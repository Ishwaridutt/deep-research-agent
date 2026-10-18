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


def _get_target_subquestions(state: WorkflowState) -> list[SubQuestion]:
    """The planner owns gap selection; this node only translates its tasks."""
    return state["research"].get("active_subquestions") or state["research"]["subquestions"]


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
    next_iteration = research["iteration_count"] + 1
    current_queries = list(research["search_queries"])
    errors = list(state["errors"])

    max_total_queries = min(
        settings.research_limits.max_total_queries_per_job,
        control.budget.max_search_queries,
    )
    remaining_query_budget = max(0, max_total_queries - max(
        len(current_queries), control.usage.queries_executed,
    ))
    if research["iteration_count"] > 0:
        remaining_query_budget = min(
            remaining_query_budget, settings.research_limits.max_followup_queries_per_round,
        )

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
    # print("\nquery generator research state:", research)
    # print("\nexisting search queries:", research["search_queries"])
    # print('LLM Service:', llm_service)

    target_subquestions = _get_target_subquestions(state)[:remaining_query_budget]
    generated_queries: list[SearchQuery] = []
    existing_queries = {_normalize_query(item.query) for item in current_queries}
    # for each subquestion generate queries
    for index, subquestion in enumerate(target_subquestions):
        if remaining_query_budget == 0:
            logger.warning("Research query budget has been exhausted.")
            break

        seconds_remaining = _seconds_remaining(state)
        if seconds_remaining <= 0:
            logger.warning("Research time budget exhausted during query generation.")
            break

        previous_queries = [item.query for item in [*current_queries, *generated_queries]]
        # Reserve one search for every remaining task; unused capacity carries forward.
        reserved_queries = len(target_subquestions) - index - 1
        max_queries = min(
            settings.research_limits.max_queries_per_subquestion,
            remaining_query_budget - reserved_queries,
        )
        if max_queries == 0:
            continue
        query_planner_prompt = build_query_generator_messages(
            original_question=state["question"],
            subquestion=subquestion,
            max_queries=max_queries,
            previous_queries=previous_queries,
            current_date=datetime.now(UTC).date().isoformat(),
        )
        metadata = LLMCallMetadata(
            job_id=state["job_id"],
            research_session_id=state["research_session_id"],
            node="research_query_generator",
            iteration=next_iteration,
            contains_private_data=ResearchSource.DOCUMENTS in state["allowed_sources"],
        )

        print('\query_planner_prompt: ')
        print(query_planner_prompt)

        try:
            query_output = await asyncio.wait_for(
                llm_service.invoke_structured(
                    role=LLMRole.QUERY_GENERATOR,
                    messages=query_planner_prompt,
                    output_schema=QueryGeneratorOutput,
                    metadata=metadata,
                ),
                timeout=seconds_remaining,
            )
        except TimeoutError:
            logger.warning("Query generation stopped at the research time limit.")
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
            if remaining_query_budget == 0 or accepted_for_subquestion >= max_queries:
                break

            query = generated.query.strip()
            purpose = generated.purpose.strip()
            if not query or not purpose:
                continue
            if len(query) > settings.research_limits.max_search_query_chars:
                continue

            normalized = _normalize_query(query)
            if normalized in existing_queries:
                continue

            existing_queries.add(normalized)
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

    if _seconds_remaining(state) <= 0:
        return {
            "research": {
                **research, "search_queries": [*current_queries, *generated_queries],
                "stop_reason": ResearchStopReason.TIME_BUDGET_EXHAUSTED,
            },
            "research_control": control.model_copy(update={
                "last_decision": ResearchDecision(
                    action="write_report", reason="Research time budget exhausted.",
                    has_unresolved_gaps=bool(control.pending_gaps), budget_exhausted=True,
                ),
            }),
            "errors": errors,
        }

    if not generated_queries:
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
        return {
            "research": {**research, "stop_reason": ResearchStopReason.NO_ACTIONABLE_GAPS},
            "research_control": control.model_copy(update={
                "last_decision": ResearchDecision(
                    action="write_report", reason="No usable new queries remain.",
                    has_unresolved_gaps=bool(control.pending_gaps),
                ),
            }),
            "errors": errors,
        }

    updated_queries = [*current_queries, *generated_queries]
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

    return {
        "research": {**research, "search_queries": updated_queries},
        "errors": errors,
    }
