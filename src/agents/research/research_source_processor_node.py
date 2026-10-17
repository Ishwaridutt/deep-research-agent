"""Research Source Processor node."""

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from pprint import pprint

from src.graph.graph_state import WorkflowState
from src.guardrails import URLValidationError, url_dedup_key, validate_public_http_url
from src.models import ErrorType, ResearchError, SearchQuery, SearchResult, Source, SourceType
from src.services import generate_source_id
from src.tools.cleaning import SourceCleaner, SourceCleaningError
from src.tools.extraction import SourceContentExtractor, SourceExtractionError


logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9]{2,}")
_STOPWORDS = {
    "about", "and", "for", "from", "how", "of", "on", "or", "the", "to",
    "what", "when", "which", "who", "why", "with",
}


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


def _tokens(value: str) -> set[str]:
    return {
        token for token in _TOKEN.findall(value.casefold())
        if token not in _STOPWORDS
    }


def _result_priority(result: SearchResult, query: SearchQuery) -> float:
    query_tokens = _tokens(query.query)
    overlap = len(query_tokens & _tokens(f"{result.title} {result.snippet or ''}"))
    relevance = overlap / len(query_tokens) if query_tokens else 0.0
    provenance = result.provider_metadata.get("discovery_provenance", [])
    provider_scores = [result.provider_score, *(
        item.get("provider_score") for item in provenance if isinstance(item, dict)
    )]
    provider_score = max(
        (score for score in provider_scores if isinstance(score, (int, float))),
        default=0.0,
    )
    return relevance + provider_score + (0.2 * min(len(provenance), 3))


def _rank_grouped_results(
    grouped_results: dict[str, list[SearchResult]],
    *,
    queries: list[SearchQuery],
    subquestion_ids: list[str],
) -> list[tuple[str, list[SearchResult]]]:
    """Rank candidates within each subquestion, then interleave the buckets."""
    queries_by_id = {query.id: query for query in queries}
    buckets: dict[str, list[tuple[float, str, list[SearchResult]]]] = {}

    for url_key, results in grouped_results.items():
        ranked_matches = [
            (_result_priority(result, query), query.subquestion_id)
            for result in results
            if (query := queries_by_id.get(result.query_id)) is not None
        ]
        priority, subquestion_id = max(ranked_matches, default=(0.0, "unassigned"))
        buckets.setdefault(subquestion_id, []).append((priority, url_key, results))

    for bucket in buckets.values():
        bucket.sort(key=lambda item: item[0], reverse=True)

    bucket_order = [item for item in subquestion_ids if item in buckets]
    bucket_order.extend(item for item in buckets if item not in bucket_order)
    ranked: list[tuple[str, list[SearchResult]]] = []
    while any(buckets[item] for item in bucket_order):
        for subquestion_id in bucket_order:
            if buckets[subquestion_id]:
                _, url_key, results = buckets[subquestion_id].pop(0)
                ranked.append((url_key, results))
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
        subquestion_ids=[item.id for item in research["subquestions"]],
    )
    sources = list(research["sources"])
    errors = list(state["errors"])
    page_attempts = control.usage.page_extraction_attempts
    round_page_attempts = 0
    skipped_for_budget = 0

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
        representative_result = search_results[0]
        search_result_ids = [result.id for result in search_results]
        canonical_key = _canonical_key(representative_result.canonical_id)

        # Reuse previously extracted pages without spending extraction budget.
        existing_index = existing_source_indexes.get(canonical_key)
        if existing_index is None:
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
        },
    )
    return {
        "research": {**research, "sources": sources},
        "research_control": updated_control,
        "errors": errors,
    }

