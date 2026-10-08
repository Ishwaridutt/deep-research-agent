"""Research Query Generator node."""

import logging
import json

from src.config import Settings
from src.graph.graph_state import WorkflowState
from src.llm import (
    LLMCallMetadata,
    LLMRole,
    LLMService,
    LLMServiceError,
)
from src.models import (
    CoverageGap,
    ErrorType,
    QueryGeneratorOutput,
    ResearchError,
    SearchQuery,
    SubQuestion,
)
from src.prompts.query_generator_prompt import (
    build_query_generator_messages,
)
from src.services import generate_search_query_id


logger = logging.getLogger(__name__)


def _normalize_query(
    value: str,
) -> str:
    """
    Normalizes a search query for deterministic duplicate detection.
    """
    return " ".join(value.strip().lower().split())


def _get_coverage_gap(
    *,
    state: WorkflowState,
    subquestion_id: str,
) -> CoverageGap | None:
    """
    Returns the current coverage gap for one subquestion.
    """

    coverage = state["research"]["coverage"]

    if coverage is None:
        return None

    for gap in coverage.gaps:
        if gap.subquestion_id == subquestion_id:
            return gap

    return None


def _get_target_subquestions(
    state: WorkflowState,
) -> list[SubQuestion]:
    """
    Determines which subquestions require queries.

    Iteration 1 targets all subquestions.
    Later iterations target only unresolved coverage gaps.
    """

    subquestions = state["research"]["subquestions"]

    if state["research"]["iteration_count"] == 0:
        return subquestions

    coverage = state["research"]["coverage"]
    if coverage is None:
        return subquestions

    gap_subquestion_ids = {
        gap.subquestion_id
        for gap in coverage.gaps
    }

    return [
        subquestion
        for subquestion in subquestions
        if (
            subquestion.id
            in gap_subquestion_ids
        )
    ]


def _get_previous_queries(
    *,
    state: WorkflowState,
    subquestion_id: str,
) -> list[str]:
    """
    Returns queries previously attempted for a subquestion.
    """

    return [
        search_query.query
        for search_query
        in state["research"]["search_queries"]
        if (
            search_query.subquestion_id
            == subquestion_id
        )
    ]


async def research_query_generator_node(
    state: WorkflowState,
    *,
    llm_service: LLMService,
    settings: Settings,
) -> dict:
    """
    Generates validated search queries for the next research iteration.
    What should we actually type into the search engine to answer each subquestion?

    Initial research targets all subquestions.
    Follow-up research targets unresolved coverage gaps.
    """

    next_iteration = state["research"]["iteration_count"] + 1
    current_queries = list(state["research"]["search_queries"])
    max_total_queries = settings.research_limits.max_total_queries_per_job
    remaining_query_budget = max_total_queries - len(current_queries)

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
    
    print("\n graph state in query gen research:", state["research"])
    print("\n graph state in query gen research:", state["research"]["search_queries"])

    if remaining_query_budget <= 0:
        logger.warning(
            "Research query budget has been exhausted.",
            extra={
                "event": "query_budget_exhausted",
                "job_id": state["job_id"],
                "research_session_id": state["research_session_id"],
                "node": ("research_query_generator"),
                "iteration": next_iteration,
            },
        )
        return {}

    target_subquestions = _get_target_subquestions(state)
    generated_queries: list[SearchQuery] = []
    existing_normalized_queries = {
        _normalize_query(search_query.query)
        for search_query in current_queries
    }

    errors = list(state["errors"])

    for subquestion in target_subquestions:
        if remaining_query_budget <= 0:
            break

        coverage_gap = _get_coverage_gap(
            state=state,
            subquestion_id=subquestion.id,
        )

        previous_queries = (
            _get_previous_queries(
                state=state,
                subquestion_id=(subquestion.id),
            )
        )

        query_generator_prompt = (
            build_query_generator_messages(
                original_question=(state["question"]),
                subquestion=subquestion,
                max_queries=settings.research_limits.max_queries_per_subquestion,
                previous_queries=previous_queries,
                coverage_gap=(
                    coverage_gap.description
                    if coverage_gap
                    else None
                ),
            )
        )

        metadata = LLMCallMetadata(
            job_id=state["job_id"],
            research_session_id=state["research_session_id"],
            node=("research_query_generator"),
            iteration=next_iteration,
        )

        try:
            query_output = await llm_service.invoke_structured(
                    role=LLMRole.QUERY_GENERATOR,
                    messages=query_generator_prompt,
                    output_schema=QueryGeneratorOutput,
                    metadata=metadata,
                )
            

        except LLMServiceError as exc:
            logger.exception(
                (
                    "Query generation failed "
                    "for subquestion."
                ),
                extra={
                    "event": (
                        "query_generation_failed"
                    ),
                    "job_id": state["job_id"],
                    "research_session_id": state["research_session_id"],
                    "node": (
                        "research_query_generator"
                    ),
                    "iteration": (
                        next_iteration
                    ),
                    "error_type": (
                        ErrorType.LLM.value
                    ),
                },
            )

            errors.append(
                ResearchError(
                    error_type=ErrorType.LLM,
                    component=(
                        "research_query_generator"
                    ),
                    message=(
                        "Could not generate "
                        "queries for subquestion "
                        f"'{subquestion.id}': "
                        f"{exc}"
                    ),
                    recoverable=True,
                )
            )

            continue

        accepted_for_subquestion = 0
        print('\ngenerated queries:', query_output.model_dump_json(indent=2))

        # validate the generated query
        for generated_query in (query_output.queries):
            if remaining_query_budget <= 0:
                break

            if (accepted_for_subquestion >= settings.research_limits.max_queries_per_subquestion):
                break

            query = generated_query.query.strip()
            purpose = generated_query.purpose.strip()

            if not query or not purpose:
                continue

            if (len(query) > settings.research_limits.max_search_query_chars):
                continue

            normalized_query = _normalize_query(query)
            # if query already exist, do not consider it
            if (normalized_query in existing_normalized_queries):
                continue
            # add normalized query into the exisiting query list
            existing_normalized_queries.add(normalized_query)

            generated_queries.append(
                SearchQuery(
                    id=generate_search_query_id(),
                    subquestion_id=subquestion.id,
                    query=query,
                    purpose=purpose,
                    # keep track in which iteration no this query was generated
                    iteration=next_iteration,
                )
            )

            accepted_for_subquestion += 1
            remaining_query_budget -= 1

    if not generated_queries:
        logger.warning(
            (
                "Query Generator produced "
                "no usable new queries."
            ),
            extra={
                "event": (
                    "no_new_queries_generated"
                ),
                "job_id": state["job_id"],
                "research_session_id": state["research_session_id"],
                "node": ("research_query_generator"),
                "iteration": next_iteration,
            },
        )

        errors.append(
            ResearchError(
                error_type=ErrorType.VALIDATION,
                component="research_query_generator",
                message=(
                    "No valid new search "
                    "queries were generated "
                    "for research iteration "
                    f"{next_iteration}."
                ),
                recoverable=True,
            )
        )

        return {
            "errors": errors,
        }

    updated_queries = [
        *current_queries,
        *generated_queries,
    ]

    logger.info(
        "Search query generation completed.",
        extra={
            "event": (
                "query_generation_completed"
            ),
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "research_query_generator",
            "iteration": next_iteration,
            # "data": updated_queries
        },
    )
    print('\nupdated_queries:',  json.dumps(
        [item.model_dump() for item in updated_queries],
        indent=2,
    ),)

    return {
        "research": {
            **state["research"],
            "search_queries": (
                updated_queries
            ),
        },
        "errors": errors,
    }


