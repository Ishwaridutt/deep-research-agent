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


def _normalize_query(value: str) -> str:
    """Normalize whitespace and case for duplicate detection."""
    return " ".join(value.casefold().split())


def _seconds_remaining(state: WorkflowState) -> float:
    control = state["research_control"]
    elapsed = (datetime.now(UTC) - control.usage.started_at).total_seconds()
    return max(0.0, control.budget.max_research_seconds - elapsed)


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
    next_iteration = research["iteration_count"] + 1
    current_queries = list(research["search_queries"])
    errors = list(state["errors"])

    max_total_queries = min(
        limits.max_total_queries_per_job,
        control.budget.max_search_queries,
    )
    remaining_query_budget = max(0, max_total_queries - max(
        len(current_queries), control.usage.queries_executed,
    ))

    logger.info(
        "\nSearch query generation started.",
        extra={
            "event": "search_query_generation_started",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "research_query_generator",
            "iteration": next_iteration,
        },
    )
    target_subquestions = (research.get("active_subquestions") or research["subquestions"])[:remaining_query_budget]
    generated_queries: list[SearchQuery] = []
    existing_queries = {_normalize_query(item.query) for item in current_queries}
    previous_queries = [item.query for item in current_queries]
    metadata = LLMCallMetadata(
        job_id=state["job_id"], research_session_id=state["research_session_id"],
        node="research_query_generator", iteration=next_iteration,
        contains_private_data=ResearchSource.DOCUMENTS in state["allowed_sources"],
    )
    timed_out = False

    # Spread the remaining job budget evenly; every selected task gets at least one query.
    max_queries = min(
        limits.max_queries_per_subquestion,
        max(1, remaining_query_budget // max(1, len(target_subquestions))),
    )
    max_query_words = getattr(limits, "max_search_query_words", 12)

    def accept(subquestion: SubQuestion, query: str, purpose: str, providers: list[str]) -> bool:
        """Validate, shorten, and deduplicate one query before it uses job budget."""
        nonlocal remaining_query_budget
        query, purpose = query.strip(), purpose.strip()
        if not query or not purpose or len(query) > limits.max_search_query_chars or remaining_query_budget <= 0:
            return False
        words = query.split()
        if len(words) > max_query_words:
            print(f"\nquery shortened to {max_query_words} words:", query)
            query = " ".join(words[:max_query_words])
        normalized = _normalize_query(query)
        if normalized in existing_queries:
            return False
        existing_queries.add(normalized)
        previous_queries.append(query)
        # Round IDs let downstream nodes process only this iteration's queries.
        generated_queries.append(SearchQuery(
            id=generate_search_query_id(),
            subquestion_id=subquestion.id,
            query=query,
            purpose=purpose,
            iteration=next_iteration,
            # General web search always runs; native branches are opt-in per query.
            providers=list(dict.fromkeys(["web", *providers])),
        ))
        remaining_query_budget -= 1
        return True

    query_output: QueryGeneratorOutput | None = None
    seconds_remaining = _seconds_remaining(state)
    if target_subquestions and seconds_remaining <= 0:
        logger.warning("Research time budget exhausted during query generation.")
        timed_out = True
    elif target_subquestions:
        # One batched call covers every task of this round.
        query_generator_prompt = build_query_generator_messages(
            original_question=state["question"],
            subquestions=target_subquestions,
            max_queries=max_queries,
            max_query_words=max_query_words,
            previous_queries=previous_queries,
            current_date=datetime.now(UTC).date().isoformat(),
        )
        print("\nquery_generator_prompt:", query_generator_prompt)
        try:
            query_output = await asyncio.wait_for(
                llm_service.invoke_structured(
                    role=LLMRole.QUERY_GENERATOR,
                    messages=query_generator_prompt,
                    output_schema=QueryGeneratorOutput,
                    metadata=metadata,
                ),
                timeout=seconds_remaining,
            )
            print("\nquery_generator_output:", query_output)
        except TimeoutError:
            logger.warning("Query generation stopped at the research time limit.")
            timed_out = True
        except LLMServiceError as exc:
            logger.exception(
                "Query generation failed; falling back to subquestion text.",
                extra={
                    "event": "query_generation_failed",
                    "job_id": state["job_id"],
                    "research_session_id": state["research_session_id"],
                    "node": "research_query_generator",
                    "iteration": next_iteration,
                    "error_type": ErrorType.LLM.value,
                },
            )
            errors.append(ResearchError(
                error_type=ErrorType.LLM,
                component="research_query_generator",
                message=f"Could not generate search queries; using subquestion text instead: {exc}",
                recoverable=True,
            ))

    if not timed_out:
        queries_by_subquestion: dict[str, list] = {}
        for group in (query_output.subquestions if query_output else []):
            queries_by_subquestion.setdefault(group.subquestion_id, []).extend(group.queries)
        unknown_ids = set(queries_by_subquestion) - {item.id for item in target_subquestions}
        if unknown_ids:
            print("\nquery generator returned unknown subquestion ids (ignored):", sorted(unknown_ids))

        fallback_subquestion_ids = []
        for subquestion in target_subquestions:
            accepted_for_subquestion = 0
            for generated in queries_by_subquestion.get(subquestion.id, []):
                if accepted_for_subquestion >= max_queries:
                    break
                if accept(subquestion, generated.query, generated.purpose, generated.providers):
                    accepted_for_subquestion += 1
            # A task the model skipped (or a failed call) still gets one search.
            if not queries_by_subquestion.get(subquestion.id) and accept(
                subquestion, subquestion.question, "Fallback search using the subquestion text.", ["web"],
            ):
                fallback_subquestion_ids.append(subquestion.id)
        if fallback_subquestion_ids:
            logger.warning(
                "Subquestion text used as fallback search query.",
                extra={
                    "event": "query_generation_fallback",
                    "job_id": state["job_id"],
                    "research_session_id": state["research_session_id"],
                    "node": "research_query_generator",
                    "iteration": next_iteration,
                    "data": {"subquestion_ids": fallback_subquestion_ids},
                },
            )

    updated_queries = [*current_queries, *generated_queries]
    stop_reason = None
    if timed_out or _seconds_remaining(state) <= 0:
        stop_reason = ResearchStopReason.TIME_BUDGET_EXHAUSTED
    elif not generated_queries and remaining_query_budget == 0:
        stop_reason = ResearchStopReason.QUERY_BUDGET_EXHAUSTED
    elif not generated_queries:
        stop_reason = ResearchStopReason.NO_ACTIONABLE_GAPS
        logger.warning(
            "Query Generator produced no usable new queries.",
            extra={
                "event": "no_new_queries_generated",
                "job_id": state["job_id"],
                "research_session_id": state["research_session_id"],
                "node": "research_query_generator",
                "iteration": next_iteration,
            },
        )
        errors.append(ResearchError(
            error_type=ErrorType.VALIDATION,
            component="research_query_generator",
            message=f"No valid new search queries were generated for research iteration {next_iteration}.",
            recoverable=True,
        ))

    result = {"research": {**research, "search_queries": updated_queries}, "errors": errors}
    # Preserve any accepted queries when stopping, without starting an empty round.
    if stop_reason is not None:
        result["research"]["stop_reason"] = stop_reason
        result["research_control"] = control.model_copy(update={
            "last_decision": ResearchDecision(
                action="write_report", reason=stop_reason.value.replace("_", " "),
                has_unresolved_gaps=bool(control.pending_gaps),
                budget_exhausted=stop_reason != ResearchStopReason.NO_ACTIONABLE_GAPS,
            ),
        })
        logger.info("Query generation stopped.", extra={
            "event": "query_generation_stopped", "job_id": state["job_id"],
            "iteration": next_iteration, "data": {"stop_reason": stop_reason.value},
        })
        return result

    logger.info(
        "Search query generation completed.",
        extra={
            "event": "query_generation_completed",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "research_query_generator",
            "iteration": next_iteration,
            "new_queries": len(generated_queries),
            "data": {
                "max_queries_per_subquestion": max_queries,
                # Which native branches the generator opted into this round.
                "provider_requests": {
                    name: sum(name in query.providers for query in generated_queries)
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
