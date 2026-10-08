"""Research Source Processor node."""

import logging
from pprint import pprint
import json

from datetime import UTC, datetime

from src.config import Settings
from src.graph.graph_state import WorkflowState
from src.guardrails import (
    URLValidationError,
    url_dedup_key,
    validate_public_http_url,
)
from src.models import (
    ErrorType,
    ResearchError,
    SearchResult,
    Source,
    SourceType,
)
from src.services import (
    generate_source_id,
)
from src.tools.extraction import (
    WebExtractionError,
    WebExtractor,
)



logger = logging.getLogger(__name__)


def _get_iteration_search_results(
    state: WorkflowState,
    *,
    iteration: int,
) -> list[SearchResult]:
    """
    Returns search results produced by queries in one research iteration.
    """

    query_ids = {
        search_query.id
        for search_query in state["research"]["search_queries"]
        if (search_query.iteration == iteration)
    }

    return [
        search_result
        for search_result in state["research"]["search_results"]
        if (search_result.query_id in query_ids)
    ]


def _group_search_results_by_url(
    search_results: list[SearchResult],
) -> dict[str, list[SearchResult]]:
    """
    Groups results by URL while preserving every SearchResult relationship.

    Invalid URLs still pass to the Source Processor where the actual URL
    guardrail produces a structured validation error.
    """

    grouped_results: dict[str, list[SearchResult]] = {}
    for search_result in search_results:
        try:
            key = url_dedup_key(search_result.url)
        except ValueError:
            key = search_result.url.strip()

        grouped_results.setdefault(key, []).append(search_result)

    return grouped_results


def _merge_search_result_ids(
    *,
    source: Source,
    search_result_ids: list[str],
) -> Source:
    """
    Returns a copy of a Source with additional discovery provenance.
    """

    merged_ids = list(
        dict.fromkeys(
            [
                *source.search_result_ids,
                *search_result_ids,
            ]
        )
    )

    return source.model_copy(
        update={
            "search_result_ids": (
                merged_ids
            )
        }
    )


async def research_source_processor_node(
    state: WorkflowState,
    *,
    web_extractor: WebExtractor,
    settings: Settings,
) -> dict:
    """
    Validates discovered URLs, extracts webpage content, and creates Sources.

    Each unique URL is extracted once while SearchResult provenance from
    multiple queries is preserved.
    """

    print('\nsource processor state:', state)

    target_iteration = state["research"]["iteration_count"] + 1
    # gather all the search result in a particular research iteration
    iteration_results = _get_iteration_search_results(
            state,
            iteration=target_iteration,
        )
    grouped_results = _group_search_results_by_url(iteration_results)
    sources = list(state["research"]["sources"])
    errors = list(state["errors"])

    existing_source_indexes = {
        url_dedup_key(source.uri): index
        for index, source
        in enumerate(sources)
    }

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

    print('\nsource processor iteration result:')
    print(
        json.dumps(
            [result.model_dump() for result in iteration_results],
            indent=2,
        )
    )
    
    print('\nsource processor grouped_results:')
    # pprint(
    #     {
    #         key: [result.model_dump() for result in results]
    #         for key, results in grouped_results.items()
    #     },
    #     width=120,
    #     sort_dicts=False,
    # )
    for url, results in grouped_results.items():
        print(f"\n{'=' * 80}")
        print(f"URL: {url}")
        print("=" * 80)

        pprint(
            [result.model_dump() for result in results],
            width=120,
            sort_dicts=False,
        )
    print('\nsource processor existing_source_indexes result:', existing_source_indexes)

    for (url_key, search_results) in grouped_results.items():
        representative_result = (
            search_results[0]
        )

        search_result_ids = [
            result.id
            for result in search_results
        ]

        # If we already extracted this URL in a previous iteration,
        # reuse its content and only extend provenance.
        existing_index = existing_source_indexes.get(url_key)
        if existing_index is not None:
            existing_source = (
                sources[existing_index]
            )

            sources[existing_index] = _merge_search_result_ids(
                    source=existing_source,
                    search_result_ids=search_result_ids,
                )
            
            continue

        try:
            validated_url = validate_public_http_url(representative_result.url)
            
        except URLValidationError as exc:
            logger.warning(
                "Search result URL rejected.",
                extra={
                    "event": (
                        "source_url_rejected"
                    ),
                    "job_id": state["job_id"],
                    "research_session_id": state["research_session_id"],
                    "node": (
                        "research_source_processor"
                    ),
                    "iteration": (
                        target_iteration
                    ),
                    "error_type": ErrorType.VALIDATION.value,
                },
            )

            errors.append(
                ResearchError(
                    error_type=(
                        ErrorType.VALIDATION
                    ),
                    component=(
                        "research_source_processor"
                    ),
                    message=(
                        "Search result URL "
                        f"was rejected: {exc}"
                    ),
                    recoverable=True,
                )
            )

            continue

        try:
            extracted_page = await web_extractor.extract(url=validated_url)
            
        except WebExtractionError as exc:
            print('source_extraction_failed', exc)
            logger.warning(
                "Webpage extraction failed.",
                extra={
                    "event": (
                        "source_extraction_failed"
                    ),
                    "job_id": state["job_id"],
                    "research_session_id": state["research_session_id"],
                    "node": (
                        "research_source_processor"
                    ),
                    "iteration": (
                        target_iteration
                    ),
                    "error_type": (
                        ErrorType
                        .EXTRACTION
                        .value
                    ),
                },
            )

            errors.append(
                ResearchError(
                    error_type=(
                        ErrorType.EXTRACTION
                    ),
                    component=(
                        "research_source_processor"
                    ),
                    message=str(exc),
                    recoverable=True,
                )
            )

            continue

        try:
            extracted_url = validate_public_http_url(extracted_page.url)

        except URLValidationError as exc:
            errors.append(
                ResearchError(
                    error_type=(
                        ErrorType.VALIDATION
                    ),
                    component=(
                        "research_source_processor"
                    ),
                    message=(
                        "Extracted source URL "
                        f"was rejected: {exc}"
                    ),
                    recoverable=True,
                )
            )

            continue

        extracted_key = url_dedup_key(extracted_url)

        # A redirect may cause this URL to resolve to a Source that
        # already exists even though the original search URL differed.
        redirected_existing_index = existing_source_indexes.get(extracted_key)
        if (redirected_existing_index is not None):
            existing_source = (
                sources[
                    redirected_existing_index
                ]
            )

            sources[redirected_existing_index] = _merge_search_result_ids(
                source=existing_source,
                search_result_ids=(
                    search_result_ids
                ),
            )

            continue
        
        # Preserve the complete extracted page so the retrieval layer can decide which content is relevant.
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

    logger.info(
        "Source processing completed.",
        extra={
            "event": (
                "source_processing_completed"
            ),
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": (
                "research_source_processor"
            ),
            "iteration": target_iteration,
        },
    )

    return {
        "research": {
            **state["research"],
            "sources": sources,
        },
        "errors": errors,
    }


