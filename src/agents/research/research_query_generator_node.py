"""Research Query Generator node."""

import asyncio
import json
import logging
from datetime import UTC, datetime

from src.config import Settings
from src.graph.graph_state import WorkflowState
from src.llm import LLMCallMetadata, LLMRole, LLMService, LLMServiceError
from src.models import (
    CoverageGap,
    ErrorType,
    QueryGeneratorOutput,
    ResearchError,
    ResearchSource,
    SearchQuery,
    SubQuestion,
)
from src.prompts.query_generator_prompt import build_query_generator_messages
from src.services import generate_search_query_id


logger = logging.getLogger(__name__)

_MAX_FOLLOWUP_EVIDENCE_ITEMS = 6
_MAX_FOLLOWUP_EVIDENCE_CHARS = 6_000


def _normalize_query(value: str) -> str:
    """Normalize whitespace and case for duplicate detection."""
    return " ".join(value.casefold().split())


def _get_active_gaps(state: WorkflowState) -> list[CoverageGap]:
    coverage = state["research"]["coverage"]
    gaps = state["research_control"].pending_gaps
    if not gaps and coverage is not None:
        gaps = coverage.gaps
    return gaps


def _get_target_subquestions(state: WorkflowState) -> list[SubQuestion]:
    """Research all subquestions initially; prioritize unresolved gaps later."""
    subquestions = state["research"]["subquestions"]
    if state["research"]["iteration_count"] == 0:
        return subquestions

    priorities = {
        item.id: max(
            (gap.priority for gap in _get_active_gaps(state) if gap.subquestion_id == item.id),
            default=0,
        )
        for item in subquestions
    }
    return sorted(
        (item for item in subquestions if priorities[item.id]),
        key=lambda item: priorities[item.id],
        reverse=True,
    )


def _get_coverage_gap(state: WorkflowState, subquestion_id: str) -> str | None:
    """Combine unresolved gap descriptions for one target."""
    descriptions = dict.fromkeys(
        gap.description.strip() for gap in _get_active_gaps(state)
        if gap.subquestion_id == subquestion_id and gap.description.strip()
    )
    return "; ".join(descriptions) or None


def _get_suggested_queries(state: WorkflowState, subquestion_id: str) -> list[str]:
    return list(dict.fromkeys(
        query.strip()
        for gap in _get_active_gaps(state)
        if gap.subquestion_id == subquestion_id
        for query in gap.suggested_queries
        if query.strip()
    ))


def _get_previous_evidence(state: WorkflowState, subquestion_id: str) -> list[str]:
    """Keep follow-up context useful without copying the whole evidence store."""
    candidates = sorted(
        (
            item for item in state["research"]["evidence"]
            if item.subquestion_id == subquestion_id
        ),
        key=lambda item: item.relevance_score,
        reverse=True,
    )[:_MAX_FOLLOWUP_EVIDENCE_ITEMS]

    selected: list[str] = []
    remaining = _MAX_FOLLOWUP_EVIDENCE_CHARS
    for item in candidates:
        title = str(item.provenance.get("title") or "Accepted evidence")
        value = f"{title}: {item.content}"[:remaining].strip()
        if not value:
            break
        selected.append(value)
        remaining -= len(value)
        if remaining == 0:
            break
    return selected


def _allocate_query_slots(
    subquestions: list[SubQuestion],
    *,
    remaining_budget: int,
    max_per_subquestion: int,
) -> dict[str, int]:
    """Allocate one query per target before assigning a second query."""
    allocations = {item.id: 0 for item in subquestions}
    for _ in range(max_per_subquestion):
        for item in subquestions:
            if remaining_budget == 0:
                return allocations
            allocations[item.id] += 1
            remaining_budget -= 1
    return allocations


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
    remaining_query_budget = max(0, max_total_queries - len(current_queries))

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

    target_subquestions = _get_target_subquestions(state)
    query_slots = _allocate_query_slots(
        target_subquestions,
        remaining_budget=remaining_query_budget,
        max_per_subquestion=settings.research_limits.max_queries_per_subquestion,
    )
    generated_queries: list[SearchQuery] = []
    existing_queries = {_normalize_query(item.query) for item in current_queries}
    # for each subquestion generate queries
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
        max_queries = query_slots[subquestion.id]
        if max_queries == 0:
            continue
        messages = build_query_generator_messages(
            original_question=state["question"],
            subquestion=subquestion,
            max_queries=max_queries,
            previous_queries=previous_queries,
            current_date=datetime.now(UTC).date().isoformat(),
            coverage_gap=_get_coverage_gap(state, subquestion.id),
            suggested_queries=_get_suggested_queries(state, subquestion.id),
            previous_evidence=_get_previous_evidence(state, subquestion.id),
        )
        metadata = LLMCallMetadata(
            job_id=state["job_id"],
            research_session_id=state["research_session_id"],
            node="research_query_generator",
            iteration=next_iteration,
            contains_private_data=ResearchSource.DOCUMENTS in state["allowed_sources"],
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
    print("\nall generated queries: ", json.dumps(
        [item.model_dump() for item in updated_queries],
        indent=2,
    ))

    return {
        "research": {**research, "search_queries": updated_queries},
        "errors": errors,
    }
