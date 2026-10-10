"""Research Query Generator node."""

import asyncio
import json
import logging
from datetime import datetime, timezone

from src.config import Settings
from src.graph.graph_state import WorkflowState
from src.llm import LLMCallMetadata, LLMRole, LLMService, LLMServiceError
from src.models import (
    CoverageGap,
    ErrorType,
    QueryGeneratorOutput,
    ResearchError,
    SearchQuery,
    SubQuestion,
)
from src.prompts.query_generator_prompt import build_query_generator_messages
from src.services import generate_search_query_id


logger = logging.getLogger(__name__)


def _normalize_query(value: str) -> str:
    """Normalize whitespace and case for duplicate detection."""
    return " ".join(value.casefold().split())


def _get_target_subquestions(state: WorkflowState) -> list[SubQuestion]:
    """Research all subquestions initially; target only unresolved gaps later."""
    subquestions = state["research"]["subquestions"]
    if state["research"]["iteration_count"] == 0:
        return subquestions

    coverage = state["research"]["coverage"]
    gaps = state["research_control"].pending_gaps
    if not gaps and coverage is not None:
        gaps = coverage.gaps

    gap_ids = {gap.subquestion_id for gap in gaps}
    return [item for item in subquestions if item.id in gap_ids]


def _get_coverage_gap(state: WorkflowState, subquestion_id: str) -> str | None:
    """Combine unresolved gaps for a target without introducing a new prompt API."""
    gaps: list[CoverageGap] = state["research_control"].pending_gaps
    if not gaps and state["research"]["coverage"] is not None:
        gaps = state["research"]["coverage"].gaps

    descriptions = dict.fromkeys(
        gap.description.strip() for gap in gaps
        if gap.subquestion_id == subquestion_id and gap.description.strip()
    )
    return "; ".join(descriptions) or None


def _seconds_remaining(state: WorkflowState) -> float:
    control = state["research_control"]
    elapsed = (datetime.now(timezone.utc) - control.usage.started_at).total_seconds()
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
    remaining_query_budget = max(0, max_total_queries - len(current_queries))

    logger.info(
        "Search query generation started.",
        extra={
            "event": "search_query_generation_started",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "research_query_generator",
            "iteration": next_iteration,
        },
    )
    print("\n graph state in query gen research:", research)
    print("\n graph state in query gen research:", research["search_queries"])

    target_subquestions = _get_target_subquestions(state)
    generated_queries: list[SearchQuery] = []
    existing_queries = {_normalize_query(item.query) for item in current_queries}

    for subquestion in target_subquestions:
        if remaining_query_budget == 0:
            logger.warning("Research query budget has been exhausted.")
            break

        seconds_remaining = _seconds_remaining(state)
        if seconds_remaining <= 0:
            logger.warning("Research time budget exhausted during query generation.")
            break

        previous_queries = [
            item.query for item in current_queries
            if item.subquestion_id == subquestion.id
        ]
        max_queries = min(
            settings.research_limits.max_queries_per_subquestion,
            remaining_query_budget,
        )
        messages = build_query_generator_messages(
            original_question=state["question"],
            subquestion=subquestion,
            max_queries=max_queries,
            previous_queries=previous_queries,
            coverage_gap=_get_coverage_gap(state, subquestion.id),
        )
        metadata = LLMCallMetadata(
            job_id=state["job_id"],
            research_session_id=state["research_session_id"],
            node="research_query_generator",
            iteration=next_iteration,
        )

        try:
            query_output = await asyncio.wait_for(
                llm_service.invoke_structured(
                    role=LLMRole.QUERY_GENERATOR,
                    messages=messages,
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

        print("\ngenerated queries:", query_output.model_dump_json(indent=2))
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
        return {"errors": errors}

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
    print("\nupdated_queries:", json.dumps([item.model_dump() for item in updated_queries], indent=2))

    return {
        "research": {**research, "search_queries": updated_queries},
        "errors": errors,
    }
