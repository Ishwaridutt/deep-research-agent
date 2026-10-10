"""Research Web Search node."""

import asyncio
import json
import logging
from datetime import datetime, timezone

from src.config import Settings
from src.graph.graph_state import WorkflowState
from src.models import ErrorType, ResearchError, SearchResult
from src.tools.search import SearchProvider, SearchProviderError


logger = logging.getLogger(__name__)


def _seconds_remaining(state: WorkflowState) -> float:
    control = state["research_control"]
    elapsed = (datetime.now(timezone.utc) - control.usage.started_at).total_seconds()
    return max(0.0, control.budget.max_research_seconds - elapsed)


async def research_web_search_node(
    state: WorkflowState,
    *,
    search_provider: SearchProvider,
    settings: Settings,
) -> dict:
    """Search the current iteration's queries without exceeding execution budgets."""
    research = state["research"]
    control = state["research_control"]
    target_iteration = research["iteration_count"] + 1
    iteration_queries = [
        query for query in research["search_queries"]
        if query.iteration == target_iteration
    ]
    current_results = list(research["search_results"])
    errors = list(state["errors"])
    new_results: list[SearchResult] = []

    # Count provider attempts, including unsuccessful calls.
    queries_executed = control.usage.queries_executed
    query_limit = min(
        settings.research_limits.max_total_queries_per_job,
        control.budget.max_search_queries,
    )

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
    print("\nresearch web search search queries:", iteration_queries)
    print("\nresearch web search search queries:", json.dumps(
        [item.model_dump() for item in iteration_queries], indent=2
    ))

    for search_query in iteration_queries:
        if queries_executed >= query_limit:
            logger.warning("Search execution budget exhausted; skipping remaining queries.")
            break

        seconds_remaining = _seconds_remaining(state)
        if seconds_remaining <= 0:
            logger.warning("Research time budget exhausted; stopping web search.")
            break

        queries_executed += 1
        try:
            results = await asyncio.wait_for(
                search_provider.search(
                    search_query=search_query,
                    max_results=settings.research_limits.max_search_results_per_query,
                ),
                timeout=seconds_remaining,
            )
        except TimeoutError:
            logger.warning("Web search timed out at the research deadline.")
            errors.append(ResearchError(
                error_type=ErrorType.SEARCH,
                component="research_web_search",
                message=f"Search timed out for query '{search_query.id}'.",
                recoverable=True,
            ))
            break
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
            errors.append(ResearchError(
                error_type=ErrorType.SEARCH,
                component="research_web_search",
                message=str(exc),
                recoverable=True,
            ))
            continue

        print("\nsearch result:", json.dumps([item.model_dump() for item in results], indent=2))
        print("\nfor query: ", search_query)

        # Retain unscored results for providers that do not expose relevance scores.
        filtered_results = [
            result for result in results
            if result.provider_score is None
            or result.provider_score >= settings.search.min_provider_score
        ]

        # Keep one fallback result rather than dropping the entire source set.
        if not filtered_results and results:
            filtered_results = [max(results, key=lambda item: item.provider_score or 0.0)]

        new_results.extend(filtered_results)

    updated_usage = control.usage.model_copy(update={"queries_executed": queries_executed})
    updated_control = control.model_copy(update={"usage": updated_usage})

    logger.info(
        "Web search completed.",
        extra={
            "event": "web_search_completed",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "research_web_search",
            "iteration": target_iteration,
            "queries_executed": queries_executed,
            "new_results": len(new_results),
        },
    )

    return {
        "research": {**research, "search_results": [*current_results, *new_results]},
        "research_control": updated_control,
        "errors": errors,
    }
