"""Research Query Generator node."""

import asyncio
import json
import logging
from datetime import UTC, datetime

from src.config import Settings
from src.graph.graph_state import WorkflowState
from src.llm import LLMCallMetadata, LLMRole, LLMService, LLMServiceError
from src.models import (
    ErrorType,
    QueryGeneratorOutput,
    ResearchError,
    ResearchSource,
    ResearchStopReason,
    SearchQuery,
    SubQuestion,
)
from src.models.research_quality import ResearchDecision
from src.prompts.query_generator_prompt import build_query_generator_messages
from src.services import generate_search_query_id


logger = logging.getLogger(__name__)

_FALLBACK_PURPOSE = "Fallback search using the subquestion text."


def _normalize_query(value: str) -> str:
    """Normalize whitespace and case for duplicate detection."""
    return " ".join(value.casefold().split())


def _seconds_remaining(state: WorkflowState) -> float:
    control = state["research_control"]
    elapsed = (datetime.now(UTC) - control.usage.started_at).total_seconds()
    return max(0.0, control.budget.max_research_seconds - elapsed)


def _remaining_query_budget(state: WorkflowState, settings: Settings) -> int:
    """Queries this job may still generate, under both the settings and the job budget."""
    control = state["research_control"]
    job_limit = min(settings.research_limits.max_total_queries_per_job, control.budget.max_search_queries)
    used = max(len(state["research"]["search_queries"]), control.usage.queries_executed)
    return max(0, job_limit - used)


def _clean_query(text: str, *, max_words: int, max_chars: int) -> str | None:
    """Trim one generated query; None when it is empty or too long to be a search phrase."""
    text = text.strip()
    if not text or len(text) > max_chars:
        return None
    words = text.split()
    if len(words) > max_words:
        print(f"\nquery shortened to {max_words} words:", text)
        text = " ".join(words[:max_words])
    return text


async def _generate_queries(
    *,
    state: WorkflowState,
    llm_service: LLMService,
    subquestions: list[SubQuestion],
    per_subquestion: int,
    max_words: int,
) -> QueryGeneratorOutput:
    """
    Ask the model for every subquestion's queries in one batched call.

    Raises TimeoutError when the research time budget runs out, and
    LLMServiceError when every configured provider fails.
    """
    seconds_remaining = _seconds_remaining(state)
    if seconds_remaining <= 0:
        raise TimeoutError
    messages = build_query_generator_messages(
        original_question=state["question"],
        subquestions=subquestions,
        max_queries=per_subquestion,
        max_query_words=max_words,
        previous_queries=[item.query for item in state["research"]["search_queries"]],
        current_date=datetime.now(UTC).date().isoformat(),
    )
    print("\nquery_generator_prompt:", messages)
    generated_queries = await asyncio.wait_for(
        llm_service.invoke_structured(
            role=LLMRole.QUERY_GENERATOR,
            messages=messages,
            output_schema=QueryGeneratorOutput,
            metadata=LLMCallMetadata(
                job_id=state["job_id"],
                research_session_id=state["research_session_id"],
                node="research_query_generator",
                iteration=state["research"]["iteration_count"] + 1,
                contains_private_data=ResearchSource.DOCUMENTS in state["allowed_sources"],
            ),
        ),
        timeout=seconds_remaining,
    )
    print("\nquery_generator_output:", generated_queries)
    return generated_queries


def _select_queries(
    *,
    subquestions: list[SubQuestion],
    generated_queries: QueryGeneratorOutput | None,
    previous_queries: list[SearchQuery],
    budget: int,
    per_subquestion: int,
    max_words: int,
    max_chars: int,
    iteration: int,
) -> tuple[list[SearchQuery], list[str]]:
    """
    Turn the model's generated queries into valid, unique SearchQuery objects within the budgets.

    A subquestion the model skipped (every subquestion, when the call failed)
    gets its own text as one web query. Returns (queries, fallback_subquestion_ids).
    """
    # The model's queries grouped by subquestion ID (empty when the call failed).
    queries_by_subquestion: dict[str, list] = {}
    for group in (generated_queries.subquestions if generated_queries else []):
        queries_by_subquestion.setdefault(group.subquestion_id, []).extend(group.queries)
    unknown_ids = set(queries_by_subquestion) - {item.id for item in subquestions}
    if unknown_ids:
        print("\nquery generator returned unknown subquestion ids (ignored):", sorted(unknown_ids))

    seen = {_normalize_query(item.query) for item in previous_queries}
    selected: list[SearchQuery] = []
    fallback_ids: list[str] = []
    for subquestion in subquestions:
        # Queries to try for this subquestion: the model's proposals, which may still be
        # rejected below (empty, too long, duplicate, or over budget).
        queries_to_try = [
            (item.query, item.purpose, item.providers)
            for item in queries_by_subquestion.get(subquestion.id, [])
        ]
        # The model proposed nothing for this subquestion (or the call failed):
        # search with the subquestion's own text instead, so it is not left without a search.
        uses_fallback = not queries_to_try
        if uses_fallback:
            queries_to_try = [(subquestion.question, _FALLBACK_PURPOSE, ["web"])]

        accepted = 0
        for text, purpose, providers in queries_to_try:
            if accepted >= per_subquestion or len(selected) >= budget:
                break
            query = _clean_query(text, max_words=max_words, max_chars=max_chars)
            if query is None or not purpose.strip() or _normalize_query(query) in seen:
                continue
            seen.add(_normalize_query(query))
            selected.append(SearchQuery(
                id=generate_search_query_id(),
                subquestion_id=subquestion.id,
                query=query,
                purpose=purpose.strip(),
                # Round IDs let downstream nodes process only this iteration's queries.
                iteration=iteration,
                # General web search always runs; native branches are opt-in per query.
                providers=list(dict.fromkeys(["web", *providers])),
            ))
            accepted += 1
        if uses_fallback and accepted:
            fallback_ids.append(subquestion.id)
    return selected, fallback_ids


def _stop_reason(
    *,
    state: WorkflowState,
    timed_out: bool,
    new_queries: list[SearchQuery],
    budget: int,
) -> ResearchStopReason | None:
    """Decide whether research must stop instead of starting a round with these queries."""
    if timed_out or _seconds_remaining(state) <= 0:
        return ResearchStopReason.TIME_BUDGET_EXHAUSTED
    if not new_queries and budget == 0:
        return ResearchStopReason.QUERY_BUDGET_EXHAUSTED
    if not new_queries:
        return ResearchStopReason.NO_ACTIONABLE_GAPS
    return None


async def research_query_generator_node(
    state: WorkflowState,
    *,
    llm_service: LLMService,
    settings: Settings,
) -> dict:
    """Generate unique queries for the next round within the job's query budget."""
    research = state["research"]
    control = state["research_control"]
    limits = settings.research_limits
    iteration = research["iteration_count"] + 1
    errors = list(state["errors"])
    log_context = {
        "job_id": state["job_id"],
        "research_session_id": state["research_session_id"],
        "node": "research_query_generator",
        "iteration": iteration,
    }
    logger.info("Search query generation started.", extra={**log_context, "event": "search_query_generation_started"})

    # 1. Budgets: every selected subquestion gets at least one query, the rest is spread evenly.
    budget = _remaining_query_budget(state, settings)
    subquestions = (research.get("active_subquestions") or research["subquestions"])[:budget]
    per_subquestion = min(limits.max_queries_per_subquestion, max(1, budget // max(1, len(subquestions))))
    max_words = getattr(limits, "max_search_query_words", 12)

    # 2. One batched LLM call for all subquestions of this round.
    generated_queries: QueryGeneratorOutput | None = None
    timed_out = False
    if subquestions:
        try:
            generated_queries = await _generate_queries(
                state=state,
                llm_service=llm_service,
                subquestions=subquestions,
                per_subquestion=per_subquestion,
                max_words=max_words,
            )
        except TimeoutError:
            logger.warning("Query generation stopped at the research time limit.")
            timed_out = True
        except LLMServiceError as exc:
            logger.exception(
                "Query generation failed; falling back to subquestion text.",
                extra={**log_context, "event": "query_generation_failed", "error_type": ErrorType.LLM.value},
            )
            errors.append(ResearchError(
                error_type=ErrorType.LLM,
                component="research_query_generator",
                message=f"Could not generate search queries; using subquestion text instead: {exc}",
                recoverable=True,
            ))

    # 3. Validate, shorten, deduplicate, and apply budgets (fallback queries when needed).
    new_queries, fallback_ids = ([], []) if timed_out else _select_queries(
        subquestions=subquestions,
        generated_queries=generated_queries,
        previous_queries=research["search_queries"],
        budget=budget,
        per_subquestion=per_subquestion,
        max_words=max_words,
        max_chars=limits.max_search_query_chars,
        iteration=iteration,
    )
    if fallback_ids:
        logger.warning(
            "Subquestion text used as fallback search query.",
            extra={**log_context, "event": "query_generation_fallback", "data": {"subquestion_ids": fallback_ids}},
        )

    updated_queries = [*research["search_queries"], *new_queries]
    result = {
        "research": {
            **research, 
            "search_queries": updated_queries
        }, 
        "errors": errors
    }

    # 4. Stop without starting an empty round; keep any accepted queries.
    stop_reason = _stop_reason(state=state, timed_out=timed_out, new_queries=new_queries, budget=budget)
    if stop_reason is not None:
        if stop_reason == ResearchStopReason.NO_ACTIONABLE_GAPS:
            logger.warning(
                "Query Generator produced no usable new queries.",
                extra={**log_context, "event": "no_new_queries_generated"},
            )
            errors.append(ResearchError(
                error_type=ErrorType.VALIDATION,
                component="research_query_generator",
                message=f"No valid new search queries were generated for research iteration {iteration}.",
                recoverable=True,
            ))
        result["research"]["stop_reason"] = stop_reason
        result["research_control"] = control.model_copy(update={
            "last_decision": ResearchDecision(
                action="write_report",
                reason=stop_reason.value.replace("_", " "),
                has_unresolved_gaps=bool(control.pending_gaps),
                budget_exhausted=stop_reason != ResearchStopReason.NO_ACTIONABLE_GAPS,
            ),
        })
        logger.info("Query generation stopped.", extra={
            **log_context, "event": "query_generation_stopped", "data": {"stop_reason": stop_reason.value},
        })
        return result

    logger.info(
        "Search query generation completed.",
        extra={
            **log_context,
            "event": "query_generation_completed",
            "new_queries": len(new_queries),
            "data": {
                "max_queries_per_subquestion": per_subquestion,
                # Which native branches the generator opted into this round.
                "provider_requests": {
                    name: sum(name in query.providers for query in new_queries)
                    for name in ("web", "arxiv", "wikipedia")
                },
            },
        },
    )
    print("\nall generated queries: ", json.dumps(
        [item.model_dump() for item in updated_queries],
        indent=2,
    ))

    return result
