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
    current_date = datetime.now(UTC).date().isoformat()
    timed_out = False
    for index, subquestion in enumerate(target_subquestions):
        seconds_remaining = _seconds_remaining(state)
        if seconds_remaining <= 0:
            logger.warning("Research time budget exhausted during query generation.")
            break

        # Reserve one search for every remaining task; unused capacity carries forward.
        reserved_queries = len(target_subquestions) - index - 1
        max_queries = min(
            limits.max_queries_per_subquestion,
            remaining_query_budget - reserved_queries,
        )
        query_generator_prompt = build_query_generator_messages(
            original_question=state["question"],
            subquestion=subquestion,
            max_queries=max_queries,
            previous_queries=previous_queries,
            current_date=current_date,
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
        except TimeoutError:
            logger.warning("Query generation stopped at the research time limit.")
            timed_out = True
            break
        except LLMServiceError as exc:
            logger.exception(
                "Query generation failed for subquestion.",
                extra={
                    "event": "query_generation_failed",
                    "job_id": state["job_id"],
                    "research_session_id": state["research_session_id"],
                    "node": "research_query_generator",
                    "iteration": next_iteration,
                    "subquestion_id": subquestion.id,
                    "error_type": ErrorType.LLM.value,
                },
            )
            errors.append(ResearchError(
                error_type=ErrorType.LLM,
                component="research_query_generator",
                message=f"Could not generate queries for subquestion '{subquestion.id}': {exc}",
                recoverable=True,
            ))
            continue

        accepted_for_subquestion = 0

        # Validate and deduplicate before adding queries to the job budget.
        for generated in query_output.queries:
            if accepted_for_subquestion >= max_queries:
                break

            query = generated.query.strip()
            purpose = generated.purpose.strip()
            if not query or not purpose or len(query) > limits.max_search_query_chars:
                continue

            normalized = _normalize_query(query)
            if normalized in existing_queries:
                continue

            existing_queries.add(normalized)
            previous_queries.append(query)
            # Round IDs let downstream nodes process only this iteration's queries.
            generated_queries.append(SearchQuery(
                id=generate_search_query_id(),
                subquestion_id=subquestion.id,
                query=query,
                purpose=purpose,
                iteration=next_iteration,
            ))
            accepted_for_subquestion += 1
            remaining_query_budget -= 1

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
        },
    )
    print("\nall generated queries: ", json.dumps(
        [item.model_dump() for item in updated_queries],
        indent=2,
    ))

    return result
