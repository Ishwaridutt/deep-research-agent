"""Research Web Search node."""

import logging
import json

from src.config import Settings
from src.graph.graph_state import WorkflowState
from src.models import (
    ErrorType,
    ResearchError,
    SearchResult,
)
from src.tools.search import (
    SearchProvider,
    SearchProviderError,
)


logger = logging.getLogger(__name__)


async def research_web_search_node(
    state: WorkflowState,
    *,
    search_provider: SearchProvider,
    settings: Settings,
) -> dict:
    """
    Executes search queries belonging to the current research iteration.

    Individual query failures are recoverable so successful queries from
    the same iteration can continue through the pipeline.
    """

    target_iteration = state["research"]["iteration_count"] + 1

    iteration_queries = [
        search_query
        for search_query
        in state["research"]["search_queries"]
        if (search_query.iteration == target_iteration)
    ]

    current_results = list(state["research"]["search_results"])
    errors = list(state["errors"])
    new_results: list[SearchResult] = []

    logger.info(
        "Web search started.",
        extra={
            "event": "web_search_started",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "research_web_search",
            "iteration": target_iteration,
        },
    )
    print('\nresearch web search search queries:', iteration_queries)
    print('\nresearch web search search queries:',  json.dumps(
            [item.model_dump() for item in iteration_queries],
            indent=2,
        ),)

    for search_query in iteration_queries:
        try:
            results = await search_provider.search(
                    search_query=search_query,
                    max_results=(settings.research_limits.max_search_results_per_query),
                )

        except SearchProviderError as exc:
            logger.exception(
                "Web search failed.",
                extra={
                    "event": "web_search_failed",
                    "job_id": state["job_id"],
                    "research_session_id": state["research_session_id"],
                    "node": "research_web_search",
                    "iteration": target_iteration,
                    "query_id": search_query.id,
                    "error_type": ErrorType.SEARCH.value,
                },
            )

            errors.append(
                ResearchError(
                    error_type=ErrorType.SEARCH,
                    component="research_web_search",
                    message=str(exc),
                    recoverable=True,
                )
            )

            continue

        # print('\nsearch result:', results,)
        print('\nsearch result:',  json.dumps(
                [item.model_dump() for item in results],
                indent=2,
            ),)
        print('\nfor query: ', search_query)

        # Discard scored provider results that do not meet the minimum relevance threshold.
        # Results without a provider score are retained for compatibility with other providers.
        filtered_results = [
            result
            for result in results
            if (
                result.provider_score is None
                or result.provider_score >= settings.search.min_provider_score
            )
        ]

        # Fallback: If every scored result was rejected, keep the best available result
        # so difficult queries do not lose their entire source set.
        if not filtered_results and results:
            filtered_results = [
                max(
                    results,
                    key=lambda result: result.provider_score or 0.0,
                )
            ]

        new_results.extend(filtered_results)

    updated_results = [
        *current_results,
        *new_results,
    ]

    # updated_results = _deduplicate_search_results(
        #     [
        #         *current_results,
        #         *new_results,
        #     ]
        # )

    logger.info(
        "Web search completed.",
        extra={
            "event": "web_search_completed",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "research_web_search",
            "iteration": target_iteration,
        },
    )

    return {
        "research": {
            **state["research"],
            "search_results": (
                updated_results
            ),
        },
        "errors": errors,
    }


