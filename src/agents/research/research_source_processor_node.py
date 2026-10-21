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
from src.tools.search.parallel_search import preferred_full_text_result, title_identity_key
from src.tools.search.relevance_gate import GateCandidate, GateScore, SearchRelevanceGate, fair_order


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


def _merge_groups_by_title(grouped: dict[str, list[SearchResult]]) -> dict[str, list[SearchResult]]:
    """Merge URL groups that are copies of one work on different hosts (e.g. PubMed and PMC)."""
    merged: dict[str, list[SearchResult]] = {}
    key_by_title: dict[str, str] = {}
    for url_key, results in grouped.items():
        title_key = next(filter(None, (title_identity_key(result.title) for result in results)), None)
        target = key_by_title.get(title_key) if title_key else None
        if target is None:
            merged[url_key] = list(results)
            if title_key:
                key_by_title[title_key] = url_key
            continue
        print(f"\nmerged duplicate copy by title: {url_key} -> {target}")
        merged[target].extend(results)
    return merged


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
    """Fallback order without a relevance gate: balance queries within and across subquestions."""
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
    for query in query_order:
        candidates = [result for results in grouped_results.values() for result in results if result.query_id == query.id]
        buckets.append([
            (results_by_id[result.id], grouped_results[results_by_id[result.id]])
            for result in sorted(candidates, key=lambda item: -(item.provider_score or 0.0))
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


def _representative_result(search_results: list[SearchResult]) -> SearchResult:
    """Prefer native metadata so a paper or article uses its dedicated extractor."""
    return next(
        (item for item in search_results if item.provider in {SourceProvider.ARXIV, SourceProvider.WIKIPEDIA}),
        preferred_full_text_result(search_results),
    )


def _print_gate_scores(
    scores: list[GateScore],
    candidates: dict[str, list[SearchResult]],
    subquestions: dict[str, str],
    threshold: float,
) -> None:
    """Print every gate decision so the threshold can be calibrated from real runs."""
    print(f"\nsource relevance gate (threshold={threshold}):")
    by_subquestion: dict[str, list[GateScore]] = {}
    for item in scores:
        by_subquestion.setdefault(item.subquestion_id, []).append(item)
    for subquestion_id, items in by_subquestion.items():
        print(f"\n  subquestion: {subquestions.get(subquestion_id, subquestion_id)}")
        for item in sorted(items, key=lambda value: -value.score):
            result = _representative_result(candidates[item.key])
            print(f"    {item.decision:<8} {item.score:.3f} [{result.provider.value}] {result.title[:100]} | {result.url}")


async def research_source_processor_node(
    state: WorkflowState,
    *,
    source_extractor: SourceContentExtractor,
    source_cleaner: SourceCleaner,
    relevance_gate: SearchRelevanceGate | None = None,
    extraction_concurrency: int = 5,
) -> dict:
    """Gate candidates by relevance, extract unique pages in parallel, and retain source provenance."""
    # print("\nsource processor state:", state)

    research = state["research"]
    control = state["research_control"]
    target_iteration = research["iteration_count"] + 1
    iteration_results = _get_iteration_search_results(state, iteration=target_iteration)
    # Candidates that passed the gate last round but missed the page budget get another chance.
    iteration_result_ids = {result.id for result in iteration_results}
    carried_ids = set(research.get("deferred_search_result_ids") or []) - iteration_result_ids
    carried_results = [result for result in research["search_results"] if result.id in carried_ids]
    grouped_results = _merge_groups_by_title(
        _group_search_results_by_url([*iteration_results, *carried_results])
    )
    active_subquestions = research.get("active_subquestions") or research["subquestions"]

    sources = list(research["sources"])
    errors = list(state["errors"])
    page_attempts = control.usage.page_extraction_attempts
    round_page_attempts = 0
    extracted_counts: dict[str, int] = {}
    reused_count = 0
    existing_source_indexes = {
        url_dedup_key(source.uri): index for index, source in enumerate(sources)
    }
    existing_source_indexes.update({
        _canonical_key(source.canonical_id): index
        for index, source in enumerate(sources)
    })
    # Copies of an already extracted work on another host are reused, not re-extracted.
    existing_source_indexes.update({
        title_key: index
        for index, source in enumerate(sources)
        if (title_key := title_identity_key(source.title))
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

    log_context = {
        "job_id": state["job_id"],
        "research_session_id": state["research_session_id"],
        "node": "research_source_processor",
        "iteration": target_iteration,
    }
    logger.info("Source processing started.", extra={**log_context, "event": "source_processing_started"})

    # print("\nsource processor iteration results:")
    # print(json.dumps(
    #     [result.model_dump(mode="json") for result in iteration_results],
    #     indent=2,
    # ))
    print("\nsource processor grouped results: ")
    print("total grouped result key: ", len(grouped_results))
    total_results = sum(len(items) for items in grouped_results.values())
    print("total grouped result items: ", total_results)
    print("carried over candidates from previous round: ", len(carried_results))
    for url, results in grouped_results.items():
        print(f"{'=' * 80}\nURL: {url}\n{'=' * 80}")
        # pprint(
        #     [result.model_dump(mode="json") for result in results],
        #     width=120,
        #     sort_dicts=False,
        # )

    print("\nsource processor existing source indexes:", existing_source_indexes)

    # 1) Reuse pages extracted in earlier rounds without spending extraction budget.
    new_candidates: dict[str, list[SearchResult]] = {}
    for url_key, search_results in grouped_results.items():
        representative_result = _representative_result(search_results)
        existing_index = existing_source_indexes.get(_canonical_key(representative_result.canonical_id))
        if existing_index is None:
            existing_index = existing_source_indexes.get(url_key)
        if existing_index is None:
            title_indexes = [
                existing_source_indexes[title_key]
                for result in search_results
                if (title_key := title_identity_key(result.title)) in existing_source_indexes
            ]
            existing_index = title_indexes[0] if title_indexes else None
        if existing_index is not None:
            reused_count += 1
            sources[existing_index] = _merge_search_result_ids(
                source=sources[existing_index],
                search_result_ids=[result.id for result in search_results],
            )
            continue
        new_candidates[url_key] = search_results

    # 2) Order new candidates: relevance gate when available, deterministic fallback otherwise.
    gate_stats: dict = {}
    if relevance_gate is not None and new_candidates:
        query_subquestions = {query.id: query.subquestion_id for query in research["search_queries"]}
        query_texts = {query.id: query.query for query in research["search_queries"]}
        subquestion_texts = {item.id: item.question for item in active_subquestions}
        gate_candidates = []
        for url_key, search_results in new_candidates.items():
            representative_result = _representative_result(search_results)
            gate_candidates.append(GateCandidate(
                key=url_key,
                text=f"{representative_result.title}\n{representative_result.snippet or ''}".strip(),
                # Carried results are matched against every active subquestion.
                subquestion_ids=tuple(dict.fromkeys(
                    query_subquestions[result.query_id] for result in search_results
                    if result.id not in carried_ids and result.query_id in query_subquestions
                )),
                query_texts=tuple(dict.fromkeys(
                    query_texts[result.query_id] for result in search_results
                    if result.id not in carried_ids and result.query_id in query_texts
                )),
            ))
        scores = await asyncio.to_thread(
            relevance_gate.evaluate, subquestions=subquestion_texts, candidates=gate_candidates,
        )
        _print_gate_scores(scores, new_candidates, subquestion_texts, relevance_gate.threshold)
        ordered_keys = [item.key for item in fair_order(scores, list(subquestion_texts))]
        decisions = [item.decision for item in scores]
        gate_stats = {
            "gate_passed_count": decisions.count("passed"),
            "gate_kept_min_count": decisions.count("kept_min"),
            "gate_rejected_count": decisions.count("rejected"),
        }
        logger.info(
            "Source relevance gate completed.",
            extra={
                **log_context,
                "event": "source_relevance_gate_completed",
                "data": {
                    "threshold": relevance_gate.threshold,
                    "min_keep": relevance_gate.min_keep,
                    "candidate_count": len(scores),
                    "carried_over_count": len(carried_results),
                    **gate_stats,
                    "by_subquestion": {
                        subquestion_id: {
                            "passed": sum(item.subquestion_id == subquestion_id and item.decision != "rejected" for item in scores),
                            "rejected": sum(item.subquestion_id == subquestion_id and item.decision == "rejected" for item in scores),
                            "top_score": max((item.score for item in scores if item.subquestion_id == subquestion_id), default=None),
                        }
                        for subquestion_id in subquestion_texts
                    },
                    # Bounded per-candidate scores for threshold calibration.
                    "scores": [
                        {
                            "subquestion_id": item.subquestion_id,
                            "score": item.score,
                            "decision": item.decision,
                            "provider": _representative_result(new_candidates[item.key]).provider.value,
                            "title": _representative_result(new_candidates[item.key]).title[:150],
                            "url": _representative_result(new_candidates[item.key]).url[:300],
                        }
                        for item in sorted(scores, key=lambda value: -value.score)[:100]
                    ],
                },
            },
        )
    else:
        ordered_keys = [
            key for key, _ in _rank_grouped_results(
                new_candidates,
                queries=research["search_queries"],
                subquestion_ids=[item.id for item in active_subquestions],
            )
        ]

    # 3) Select pages within the job, round, and time budgets.
    selected: list[tuple[str, list[SearchResult], SearchResult, str]] = []
    deferred_keys: list[str] = []
    for url_key in ordered_keys:
        search_results = new_candidates[url_key]
        representative_result = _representative_result(search_results)
        try:
            validated_url = validate_public_http_url(representative_result.url)
        except URLValidationError as exc:
            logger.warning(
                "Search result URL rejected.",
                extra={**log_context, "event": "source_url_rejected", "error_type": ErrorType.VALIDATION.value},
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
            deferred_keys.append(url_key)
            continue

        if _seconds_remaining(state) <= 0:
            logger.warning("Research time budget exhausted; skipping new URLs.")
            continue

        # Count extraction calls against the page budget.
        page_attempts += 1
        round_page_attempts += 1
        selected.append((url_key, search_results, representative_result, validated_url))

    # 4) Extract the selected pages concurrently within a bounded limit.
    semaphore = asyncio.Semaphore(max(1, extraction_concurrency))

    async def extract_page(representative_result: SearchResult, validated_url: str):
        """Extract one page; failures are returned so the round can continue."""
        async with semaphore:
            seconds_remaining = _seconds_remaining(state)
            if seconds_remaining <= 0:
                return TimeoutError()
            try:
                return await asyncio.wait_for(
                    source_extractor.extract(search_result=representative_result.model_copy(
                        update={"url": validated_url},
                    )),
                    timeout=seconds_remaining,
                )
            except (TimeoutError, SourceExtractionError) as exc:
                return exc

    print(f"\nextracting {len(selected)} pages with concurrency={max(1, extraction_concurrency)}")
    outcomes = await asyncio.gather(*(
        extract_page(representative_result, validated_url)
        for _, _, representative_result, validated_url in selected
    ))

    # 5) Process extracted pages in ranked order so deduplication stays deterministic.
    for (url_key, search_results, representative_result, validated_url), extracted_source in zip(selected, outcomes):
        search_result_ids = [result.id for result in search_results]
        canonical_key = _canonical_key(representative_result.canonical_id)

        if isinstance(extracted_source, TimeoutError):
            logger.warning("Page extraction timed out at the research deadline.")
            errors.append(ResearchError(
                error_type=ErrorType.EXTRACTION,
                component="research_source_processor",
                message=f"Page extraction timed out for URL '{validated_url}'.",
                recoverable=True,
            ))
            continue

        if isinstance(extracted_source, SourceExtractionError):
            print("source extraction failed:", extracted_source)
            logger.warning(
                "Webpage extraction failed.",
                extra={**log_context, "event": "source_extraction_failed", "error_type": ErrorType.EXTRACTION.value},
            )
            errors.append(ResearchError(
                error_type=ErrorType.EXTRACTION,
                component="research_source_processor",
                message=str(extracted_source),
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
        if title_key := title_identity_key(source.title):
            existing_source_indexes[title_key] = len(sources) - 1

    skipped_for_budget = len(deferred_keys)
    deferred_search_result_ids = [
        result.id for url_key in deferred_keys for result in new_candidates[url_key]
    ]
    if skipped_for_budget:
        logger.info(
            "Source candidates were deferred after the round extraction limit.",
            extra={
                **log_context,
                "event": "source_candidates_deferred",
                "skipped_count": skipped_for_budget,
                "round_page_attempts": round_page_attempts,
            },
        )

    updated_usage = control.usage.model_copy(update={"page_extraction_attempts": page_attempts})
    updated_control = control.model_copy(update={"usage": updated_usage})
    extraction_stats = {
        "iteration": target_iteration,
        "extracted_provider_counts": extracted_counts,
        "extracted_count": sum(extracted_counts.values()),
        "reused_count": reused_count,
        "attempted_count": round_page_attempts,
        "deferred_count": skipped_for_budget,
        "carried_over_count": len(carried_results),
        **gate_stats,
    }

    logger.info(
        "Source processing completed.",
        extra={
            **log_context,
            "event": "source_processing_completed",
            "page_extraction_attempts": page_attempts,
            "round_page_attempts": round_page_attempts,
            "sources_count": len(sources),
            "data": {key: value for key, value in extraction_stats.items() if key != "iteration"},
        },
    )
    return {
        "research": {
            **research,
            "sources": sources,
            "deferred_search_result_ids": deferred_search_result_ids,
            "extraction_stats": extraction_stats,
        },
        "research_control": updated_control,
        "errors": errors,
    }
