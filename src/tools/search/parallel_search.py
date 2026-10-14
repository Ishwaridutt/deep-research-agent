"""Parallel public search composition and canonical deduplication."""

import asyncio
import re
from urllib.parse import unquote, urlsplit

from src.guardrails import url_dedup_key
from src.models import SearchQuery, SearchResult, SourceProvider
from src.tools.search.base import SearchProvider, SearchProviderError


_ARXIV_VERSION = re.compile(r"v\d+$", re.IGNORECASE)


class ParallelSearchProvider(SearchProvider):
    """Runs independent search branches concurrently and merges duplicates."""

    def __init__(
        self,
        *,
        branches: list[tuple[str, SearchProvider, int]],
        max_concurrency: int,
    ) -> None:
        if not branches:
            raise ValueError("At least one search branch is required.")
        self._branches = branches
        self._semaphore = asyncio.Semaphore(max_concurrency)

    async def search(
        self,
        *,
        search_query: SearchQuery,
        max_results: int,
    ) -> list[SearchResult]:
        # Each branch keeps its own ranking and result cap; the semaphore limits
        # total outbound searches without forcing the providers to run sequentially.
        outcomes = await asyncio.gather(*(
            self._search_branch(
                name=name,
                provider=provider,
                search_query=search_query,
                max_results=min(limit, max_results),
            )
            for name, provider, limit in self._branches
        ))

        # A failed provider must not discard useful results returned by the others.
        results = [
            result
            for _, branch_results, _ in outcomes
            for result in branch_results
        ]
        failures = [name for name, _, error in outcomes if error is not None]
        if failures and len(failures) == len(outcomes):
            raise SearchProviderError(
                "All public search branches failed for query "
                f"'{search_query.id}': {', '.join(failures)}."
            )

        deduplicated = deduplicate_search_results(results)
        successful_branches = [
            name for name, _, error in outcomes if error is None
        ]
        diagnostics = {
            "successful_branches": successful_branches,
            "failed_branches": failures,
            "web_fallback_activated": any(
                result.provider == SourceProvider.BRAVE for result in results
            ),
        }
        # Attach only safe branch-level diagnostics so the graph can report
        # partial progress without retaining errors or provider responses.
        deduplicated = [
            result.model_copy(update={
                "provider_metadata": {
                    **result.provider_metadata,
                    "search_diagnostics": diagnostics,
                },
            })
            for result in deduplicated
        ]
        print(
            "Parallel search completed:",
            f"query={search_query.id}",
            f"results={len(deduplicated)}",
            f"failed_branches={failures}",
        )
        return deduplicated

    async def _search_branch(
        self,
        *,
        name: str,
        provider: SearchProvider,
        search_query: SearchQuery,
        max_results: int,
    ) -> tuple[str, list[SearchResult], SearchProviderError | None]:
        try:
            async with self._semaphore:
                results = await provider.search(
                    search_query=search_query,
                    max_results=max_results,
                )
            return name, results[:max_results], None
        except SearchProviderError as exc:
            return name, [], exc

    async def close(self) -> None:
        # Attempt every cleanup even if one provider fails to close.
        outcomes = await asyncio.gather(
            *(provider.close() for _, provider, _ in self._branches),
            return_exceptions=True,
        )
        if any(isinstance(outcome, Exception) for outcome in outcomes):
            raise SearchProviderError("One or more search providers failed to close.")


def deduplicate_search_results(
    results: list[SearchResult],
) -> list[SearchResult]:
    """Merge results that identify the same page, paper, or article."""

    groups: list[tuple[set[str], list[SearchResult]]] = []
    for result in results:
        identities = _identity_keys(result)
        matches = [group for group in groups if group[0] & identities]
        if not matches:
            groups.append((identities, [result]))
            continue

        target = matches[0]
        target[0].update(identities)
        target[1].append(result)

        # DOI, arXiv ID, page ID, and URL aliases can reveal that two earlier
        # groups describe the same underlying source.
        for duplicate in matches[1:]:
            target[0].update(duplicate[0])
            target[1].extend(duplicate[1])
            groups.remove(duplicate)

    return [_merge_group(group_results) for _, group_results in groups]


def _merge_group(results: list[SearchResult]) -> SearchResult:
    native_providers = {SourceProvider.ARXIV, SourceProvider.WIKIPEDIA}

    # Native results carry the IDs and metadata required by their native extractors.
    representative = next(
        (result for result in results if result.provider in native_providers),
        results[0],
    )
    provenance = [
        {
            "provider": result.provider.value,
            "result_id": result.id,
            "canonical_id": result.canonical_id,
            "provider_score": result.provider_score,
        }
        for result in results
    ]
    metadata = {
        **representative.provider_metadata,
        "discovery_provenance": provenance,
    }
    return representative.model_copy(update={"provider_metadata": metadata})


def _identity_keys(result: SearchResult) -> set[str]:
    # Multiple identities allow a Tavily/Brave URL to match the corresponding
    # native paper or article even when their provider canonical IDs differ.
    keys = {f"canonical:{result.canonical_id.casefold()}"}
    metadata = result.provider_metadata

    doi = _normalize_doi(str(metadata.get("doi") or ""))
    doi = doi or _doi_from_url(result.url)
    if doi:
        keys.add(f"doi:{doi}")

    arxiv_id = str(metadata.get("arxiv_id") or "").strip()
    arxiv_id = arxiv_id or _arxiv_id_from_url(result.url)
    if arxiv_id:
        keys.add(f"arxiv:{_ARXIV_VERSION.sub('', arxiv_id).casefold()}")

    page_id = metadata.get("page_id")
    if isinstance(page_id, int):
        keys.add(f"wikipedia-page:{page_id}")

    wikipedia_title = _wikipedia_title_from_url(result.url)
    if wikipedia_title:
        keys.add(f"wikipedia-title:{wikipedia_title}")

    try:
        keys.add(f"url:{url_dedup_key(result.url)}")
    except ValueError:
        pass
    return keys


def _normalize_doi(value: str) -> str:
    doi = value.strip().casefold()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
            break
    return doi


def _doi_from_url(url: str) -> str:
    parsed = urlsplit(url)
    if (parsed.hostname or "").casefold() not in {"doi.org", "dx.doi.org"}:
        return ""
    return _normalize_doi(unquote(parsed.path).strip("/"))


def _arxiv_id_from_url(url: str) -> str:
    parsed = urlsplit(url)
    if (parsed.hostname or "").casefold() not in {
        "arxiv.org",
        "www.arxiv.org",
        "export.arxiv.org",
    }:
        return ""

    path = unquote(parsed.path).strip("/")
    for prefix in ("abs/", "pdf/", "html/"):
        if path.startswith(prefix):
            paper_id = path[len(prefix):]
            return paper_id.removesuffix(".pdf")
    return ""


def _wikipedia_title_from_url(url: str) -> str:
    parsed = urlsplit(url)
    if not (parsed.hostname or "").casefold().endswith(".wikipedia.org"):
        return ""
    if not parsed.path.startswith("/wiki/"):
        return ""
    return unquote(parsed.path[6:]).replace("_", " ").strip().casefold()
