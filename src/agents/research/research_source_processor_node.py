"""Research Source Processor node."""

import asyncio
import json
import logging
from datetime import UTC, datetime
from itertools import zip_longest
from pprint import pprint

from src.graph.graph_state import WorkflowState
from src.guardrails import URLValidationError, url_dedup_key, validate_public_http_url
from src.models import ErrorType, ResearchError, SearchQuery, SearchResult, Source, SourceProvider, SourceType
from src.services import generate_source_id
from src.tools.cleaning import SourceCleaner, SourceCleaningError
from src.tools.extraction import SourceContentExtractor, SourceExtractionError
from src.tools.search.parallel_search import rank_search_results


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


def _canonical_key(canonical_id: str) -> str:
    return f"canonical:{canonical_id.casefold()}"


def _rank_grouped_results(
    grouped_results: dict[str, list[SearchResult]],
    *,
    queries: list[SearchQuery],
    subquestion_ids: list[str],
) -> list[tuple[str, list[SearchResult]]]:
    """Balance query and provider opportunities within the existing page budget."""
    task_queries: dict[str, list[SearchQuery]] = {}
    for query in queries:
        if any(result.query_id == query.id for results in grouped_results.values() for result in results):
            task_queries.setdefault(query.subquestion_id, []).append(query)
    task_order = list(dict.fromkeys([*subquestion_ids, *task_queries]))
    query_order = [
        query for group in zip_longest(*(task_queries.get(task, []) for task in task_order))
        for query in group if query is not None
    ]
    results_by_id = {result.id: key for key, results in grouped_results.items() for result in results}
    buckets = []
    for index, query in enumerate(query_order):
        candidates = [result for results in grouped_results.values() for result in results if result.query_id == query.id]
        buckets.append([
            (results_by_id[result.id], grouped_results[results_by_id[result.id]])
            for result in rank_search_results(candidates, query.query, provider_offset=index)
        ])

    # Deduplicate globally, retaining every query's discovery IDs on each page.
    ranked: list[tuple[str, list[SearchResult]]] = []
    seen: set[str] = set()
    while any(buckets):
        for bucket in buckets:
            while bucket and bucket[0][0] in seen:
                bucket.pop(0)
            if bucket:
                url_key, results = bucket.pop(0)
                seen.add(url_key)
                ranked.append((url_key, results))
    ranked.extend((key, results) for key, results in grouped_results.items() if key not in seen)
    return ranked


def _seconds_remaining(state: WorkflowState) -> float:
    control = state["research_control"]
    elapsed = (datetime.now(UTC) - control.usage.started_at).total_seconds()
    return max(0.0, control.budget.max_research_seconds - elapsed)


async def research_source_processor_node(
    state: WorkflowState,
    *,
    source_extractor: SourceContentExtractor,
    source_cleaner: SourceCleaner,
) -> dict:
    """Validate URLs, extract unique pages, extracts page content and retain source provenance."""
    # print("\nsource processor state:", state)

    research = state["research"]
    control = state["research_control"]
    target_iteration = research["iteration_count"] + 1
    iteration_results = _get_iteration_search_results(state, iteration=target_iteration)
    grouped_results = _group_search_results_by_url(iteration_results)
    ranked_results = _rank_grouped_results(
        grouped_results,
        queries=research["search_queries"],
        subquestion_ids=[item.id for item in (research.get("active_subquestions") or research["subquestions"])],
    )
    sources = list(research["sources"])
    errors = list(state["errors"])
    page_attempts = control.usage.page_extraction_attempts
    round_page_attempts = 0
    skipped_for_budget = 0
    extracted_counts: dict[str, int] = {}
    reused_count = 0

    existing_source_indexes = {
        url_dedup_key(source.uri): index for index, source in enumerate(sources)
    }
    existing_source_indexes.update({
        _canonical_key(source.canonical_id): index
        for index, source in enumerate(sources)
    })
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
    # print("\nsource processor iteration results:")
    # print(json.dumps(
    #     [result.model_dump(mode="json") for result in iteration_results],
    #     indent=2,
    # ))
    print("\nsource processor grouped results: ")
    print("total grouped result key: ", len(grouped_results))
    total_results = sum(len(items) for items in grouped_results.values())
    print("total grouped result items: ", total_results)
    for url, results in grouped_results.items():
        print(f"{'=' * 80}\nURL: {url}\n{'=' * 80}")
        # pprint(
        #     [result.model_dump(mode="json") for result in results],
        #     width=120,
        #     sort_dicts=False,
        # )
    print("\nsource processor existing source indexes:", existing_source_indexes)

    for url_key, search_results in ranked_results:
        # A shared URL can be discovered by different queries/providers. Retain
        # native metadata so its page or paper uses the correct extractor.
        representative_result = next(
            (item for item in search_results if item.provider in {SourceProvider.ARXIV, SourceProvider.WIKIPEDIA}),
            search_results[0],
        )
        search_result_ids = [result.id for result in search_results]
        canonical_key = _canonical_key(representative_result.canonical_id)

        # Reuse previously extracted pages without spending extraction budget.
        existing_index = existing_source_indexes.get(canonical_key)
        if existing_index is None:
            existing_index = existing_source_indexes.get(url_key)
        if existing_index is not None:
            reused_count += 1
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

        if (
            page_attempts >= control.budget.max_extracted_pages
            or round_page_attempts >= control.budget.max_extracted_pages_per_round
        ):
            skipped_for_budget += 1
            continue

        seconds_remaining = _seconds_remaining(state)
        if seconds_remaining <= 0:
            logger.warning("Research time budget exhausted; skipping new URLs.")
            continue

        # Count extraction calls against the page budget.
        page_attempts += 1
        round_page_attempts += 1
        try:
            extracted_source = await asyncio.wait_for(
                source_extractor.extract(search_result=representative_result.model_copy(
                    update={"url": validated_url},
                )),
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
        except SourceExtractionError as exc:
            print("source extraction failed:", exc)
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
            extracted_url = validate_public_http_url(extracted_source.url)
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
        redirected_index = existing_source_indexes.get(canonical_key)
        if redirected_index is None:
            redirected_index = existing_source_indexes.get(extracted_key)
        if redirected_index is not None:
            sources[redirected_index] = _merge_search_result_ids(
                source=sources[redirected_index], search_result_ids=search_result_ids
            )
            existing_source_indexes[url_key] = redirected_index
            existing_source_indexes[canonical_key] = redirected_index
            continue

        # clean the extracted content
        try:
            cleaned = source_cleaner.clean(
                extracted_source.content,
                content_format=extracted_source.content_format,
            )
        except SourceCleaningError as exc:
            print("source cleaning failed:", exc)
            errors.append(ResearchError(
                error_type=ErrorType.EXTRACTION,
                component="source_cleaner",
                message=str(exc),
                recoverable=True,
            ))
            continue

        print("Cleaned source:", {
            "version": cleaned.version,
            "original_chars": cleaned.original_chars,
            "cleaned_chars": cleaned.cleaned_chars,
            "truncated": cleaned.truncated,
        })

        source = Source(
            id=generate_source_id(),
            source_type=SourceType.WEB,
            provider=representative_result.provider,
            canonical_id=representative_result.canonical_id,
            search_result_ids=search_result_ids,
            title=representative_result.title,
            uri=extracted_url,
            content=cleaned.content,
            retrieved_at=datetime.now(UTC),
            provider_metadata={
                **representative_result.provider_metadata,
                **extracted_source.provider_metadata,
                "cleaner_version": cleaned.version,
                "original_char_count": cleaned.original_chars,
                "cleaned_char_count": cleaned.cleaned_chars,
                "content_truncated": cleaned.truncated,
            },
        )
        sources.append(source)
        provider = source.provider.value
        extracted_counts[provider] = extracted_counts.get(provider, 0) + 1
        existing_source_indexes[extracted_key] = len(sources) - 1
        existing_source_indexes[url_key] = len(sources) - 1
        existing_source_indexes[canonical_key] = len(sources) - 1

    if skipped_for_budget:
        logger.info(
            "Source candidates were deferred after the round extraction limit.",
            extra={
                "event": "source_candidates_deferred",
                "job_id": state["job_id"],
                "research_session_id": state["research_session_id"],
                "node": "research_source_processor",
                "iteration": target_iteration,
                "skipped_count": skipped_for_budget,
                "round_page_attempts": round_page_attempts,
            },
        )

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
            "round_page_attempts": round_page_attempts,
            "sources_count": len(sources),
            "data": {"extracted_provider_counts": extracted_counts, "extracted_count": sum(extracted_counts.values()),
                     "reused_count": reused_count, "attempted_count": round_page_attempts,
                     "deferred_count": skipped_for_budget},
        },
    )
    return {
        "research": {
            **research, "sources": sources,
            "extraction_stats": {
                "iteration": target_iteration,
                "extracted_provider_counts": extracted_counts,
                "extracted_count": sum(extracted_counts.values()),
                "reused_count": reused_count,
                "attempted_count": round_page_attempts,
                "deferred_count": skipped_for_budget,
            },
        },
        "research_control": updated_control,
        "errors": errors,
    }

