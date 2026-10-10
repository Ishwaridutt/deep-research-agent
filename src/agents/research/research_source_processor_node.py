"""Research Source Processor node."""

import asyncio
import json
import logging
from datetime import UTC, datetime
from pprint import pprint

from src.config import Settings
from src.graph.graph_state import WorkflowState
from src.guardrails import URLValidationError, url_dedup_key, validate_public_http_url
from src.models import ErrorType, ResearchError, SearchResult, Source, SourceType
from src.services import generate_source_id
from src.tools.extraction import WebExtractionError, WebExtractor


logger = logging.getLogger(__name__)


def _get_iteration_search_results(state: WorkflowState, *, iteration: int) -> list[SearchResult]:
    """Get search results associated with queries from the current round."""
    query_ids = {
        query.id for query in state["research"]["search_queries"]
        if query.iteration == iteration
    }
    return [result for result in state["research"]["search_results"] if result.query_id in query_ids]


def _group_search_results_by_url(search_results: list[SearchResult]) -> dict[str, list[SearchResult]]:
    """Deduplicate URLs while preserving every discovery relationship."""
    grouped: dict[str, list[SearchResult]] = {}
    for result in search_results:
        try:
            key = url_dedup_key(result.url)
        except ValueError:
            # Let the URL guardrail produce the authoritative validation error.
            key = result.url.strip()
        grouped.setdefault(key, []).append(result)
    return grouped


def _merge_search_result_ids(*, source: Source, search_result_ids: list[str]) -> Source:
    """Extend an existing source's search discovery provenance."""
    merged = list(dict.fromkeys([*source.search_result_ids, *search_result_ids]))
    return source.model_copy(update={"search_result_ids": merged})


def _seconds_remaining(state: WorkflowState) -> float:
    control = state["research_control"]
    elapsed = (datetime.now(UTC) - control.usage.started_at).total_seconds()
    return max(0.0, control.budget.max_research_seconds - elapsed)


async def research_source_processor_node(
    state: WorkflowState,
    *,
    web_extractor: WebExtractor,
    settings: Settings,
) -> dict:
    """Validate URLs, extract unique pages, and retain source provenance."""
    print("\nsource processor state:", state)

    research = state["research"]
    control = state["research_control"]
    target_iteration = research["iteration_count"] + 1
    iteration_results = _get_iteration_search_results(state, iteration=target_iteration)
    grouped_results = _group_search_results_by_url(iteration_results)
    sources = list(research["sources"])
    errors = list(state["errors"])
    page_attempts = control.usage.page_extraction_attempts

    existing_source_indexes = {
        url_dedup_key(source.uri): index for index, source in enumerate(sources)
    }
    # Index original discovery URLs too, so redirects remain reusable across rounds.
    results_by_id = {result.id: result for result in research["search_results"]}
    for index, source in enumerate(sources):
        for result_id in source.search_result_ids:
            result = results_by_id.get(result_id)
            if result is not None:
                try:
                    existing_source_indexes.setdefault(url_dedup_key(result.url), index)
                except ValueError:
                    pass

    logger.info(
        "Source processing started.",
        extra={
            "event": "source_processing_started",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "research_source_processor",
            "iteration": target_iteration,
        },
    )
    print("\nsource processor iteration result:")
    print(json.dumps([result.model_dump() for result in iteration_results], indent=2))
    print("\nsource processor grouped_results:")
    for url, results in grouped_results.items():
        print(f"\n{'=' * 80}")
        print(f"URL: {url}")
        print("=" * 80)
        pprint([result.model_dump() for result in results], width=120, sort_dicts=False)
    print("\nsource processor existing_source_indexes result:", existing_source_indexes)

    for url_key, search_results in grouped_results.items():
        representative_result = search_results[0]
        search_result_ids = [result.id for result in search_results]

        # Reuse previously extracted pages without spending extraction budget.
        existing_index = existing_source_indexes.get(url_key)
        if existing_index is not None:
            sources[existing_index] = _merge_search_result_ids(
                source=sources[existing_index], search_result_ids=search_result_ids
            )
            continue

        try:
            validated_url = validate_public_http_url(representative_result.url)
        except URLValidationError as exc:
            logger.warning(
                "Search result URL rejected.",
                extra={
                    "event": "source_url_rejected",
                    "job_id": state["job_id"],
                    "research_session_id": state["research_session_id"],
                    "node": "research_source_processor",
                    "iteration": target_iteration,
                    "error_type": ErrorType.VALIDATION.value,
                },
            )
            errors.append(ResearchError(
                error_type=ErrorType.VALIDATION,
                component="research_source_processor",
                message=f"Search result URL was rejected: {exc}",
                recoverable=True,
            ))
            continue

        if page_attempts >= control.budget.max_extracted_pages:
            logger.warning("Page extraction budget exhausted; skipping new URLs.")
            continue

        seconds_remaining = _seconds_remaining(state)
        if seconds_remaining <= 0:
            logger.warning("Research time budget exhausted; skipping new URLs.")
            continue

        # Count failed and timed-out extraction calls against the page budget.
        page_attempts += 1
        try:
            extracted_page = await asyncio.wait_for(
                web_extractor.extract(url=validated_url),
                timeout=seconds_remaining,
            )
        except TimeoutError:
            logger.warning("Page extraction timed out at the research deadline.")
            errors.append(ResearchError(
                error_type=ErrorType.EXTRACTION,
                component="research_source_processor",
                message=f"Page extraction timed out for URL '{validated_url}'.",
                recoverable=True,
            ))
            break
        except WebExtractionError as exc:
            print("source_extraction_failed", exc)
            logger.warning(
                "Webpage extraction failed.",
                extra={
                    "event": "source_extraction_failed",
                    "job_id": state["job_id"],
                    "research_session_id": state["research_session_id"],
                    "node": "research_source_processor",
                    "iteration": target_iteration,
                    "error_type": ErrorType.EXTRACTION.value,
                },
            )
            errors.append(ResearchError(
                error_type=ErrorType.EXTRACTION,
                component="research_source_processor",
                message=str(exc),
                recoverable=True,
            ))
            continue

        try:
            extracted_url = validate_public_http_url(extracted_page.url)
        except URLValidationError as exc:
            errors.append(ResearchError(
                error_type=ErrorType.VALIDATION,
                component="research_source_processor",
                message=f"Extracted source URL was rejected: {exc}",
                recoverable=True,
            ))
            continue

        extracted_key = url_dedup_key(extracted_url)

        # Redirects may resolve to a source already extracted in this or an earlier round.
        redirected_index = existing_source_indexes.get(extracted_key)
        if redirected_index is not None:
            sources[redirected_index] = _merge_search_result_ids(
                source=sources[redirected_index], search_result_ids=search_result_ids
            )
            existing_source_indexes[url_key] = redirected_index
            continue

        # Preserve full page content; retrieval will select relevant passages.
        extracted_content = extracted_page.content.strip()
        if not extracted_content:
            continue

        source = Source(
            id=generate_source_id(),
            source_type=SourceType.WEB,
            search_result_ids=search_result_ids,
            title=representative_result.title,
            uri=extracted_url,
            content=extracted_content,
            retrieved_at=datetime.now(UTC),
        )
        sources.append(source)
        existing_source_indexes[extracted_key] = len(sources) - 1
        existing_source_indexes[url_key] = len(sources) - 1

    updated_usage = control.usage.model_copy(update={"page_extraction_attempts": page_attempts})
    updated_control = control.model_copy(update={"usage": updated_usage})

    logger.info(
        "Source processing completed.",
        extra={
            "event": "source_processing_completed",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "research_source_processor",
            "iteration": target_iteration,
            "page_extraction_attempts": page_attempts,
            "sources_count": len(sources),
        },
    )
    return {
        "research": {**research, "sources": sources},
        "research_control": updated_control,
        "errors": errors,
    }

