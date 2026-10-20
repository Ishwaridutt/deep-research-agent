"""Research Web Search node."""

import asyncio
import json
import logging
from collections import Counter
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
    provider_counts: dict[str, int] = {}
    failed_branches: set[str] = set()
    fallback_activated = False
    failed_query_count = 0

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
    print("\nresearch web search queries:", json.dumps(
        [item.model_dump() for item in iteration_queries],
        indent=2,
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
            failed_query_count += 1
            logger.warning("Web search timed out at the research deadline.")
            errors.append(ResearchError(
                error_type=ErrorType.SEARCH,
                component="research_web_search",
                message=f"Search timed out for query '{search_query.id}'.",
                recoverable=True,
            ))
            break
        except SearchProviderError as exc:
            failed_query_count += 1
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

        # print("\nsearch results:", json.dumps(
        #     [item.model_dump(mode="json") for item in results],
        #     indent=2,
        # ))
        print("\nweb search completed for query:", search_query)
        print("\ntotal search results: ", len(results))

        # Capture discovery before aggregate ranking and score filtering hide providers.
        diagnostics = getattr(results, "diagnostics", {})
        counts = diagnostics.get("provider_counts")
        if counts is None:
            counts = {}
            for result in results:
                name = result.provider.value
                counts[name] = counts.get(name, 0) + 1
        for name, count in counts.items():
            provider_counts[name] = provider_counts.get(name, 0) + count
        failed_branches.update(diagnostics.get("failed_branches", []))
        fallback_activated |= bool(diagnostics.get("web_fallback_activated"))

        # Retain unscored results for providers that do not expose relevance scores.
        filtered_results = [
            result for result in results
            if result.provider_score is None
            or result.provider_score >= settings.search.min_provider_score
        ][:settings.research_limits.max_search_results_per_query]

        # Keep one fallback result rather than dropping the entire source set.
        if not filtered_results and results:
            filtered_results = [max(results, key=lambda item: item.provider_score or 0.0)]

        new_results.extend(filtered_results)

    updated_usage = control.usage.model_copy(update={"queries_executed": queries_executed})
    updated_control = control.model_copy(update={"usage": updated_usage})
    executed_query_count = queries_executed - control.usage.queries_executed

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
            "data": {
                "count_scope": "round", "executed_query_count": executed_query_count,
                "discovered_provider_counts": provider_counts, "discovered_count": sum(provider_counts.values()),
                "provider_counts": dict(Counter(result.provider.value for result in new_results)),
                "result_count": len(new_results), "failed_branches": sorted(failed_branches),
                "failed_query_count": failed_query_count,
            },
        },
    )

    return {
        "research": {
            **research, "search_results": [*current_results, *new_results],
            "search_stats": {
                "iteration": target_iteration, "provider_counts": provider_counts,
                "executed_query_count": executed_query_count,
                "failed_branches": sorted(failed_branches),
                "web_fallback_activated": fallback_activated,
                "failed_query_count": failed_query_count,
            },
        },
        "research_control": updated_control,
        "errors": errors,
    }

