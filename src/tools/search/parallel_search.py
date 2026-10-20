"""Parallel public search composition and canonical deduplication."""

import asyncio
import re
from itertools import zip_longest
from urllib.parse import unquote, urlsplit

from src.guardrails import url_dedup_key
from src.models import SearchQuery, SearchResult, SourceProvider
from src.tools.search.base import SearchProvider, SearchProviderError


_ARXIV_VERSION = re.compile(r"v\d+$", re.IGNORECASE)
_TOKEN = re.compile(r"[a-z0-9]{2,}")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_SEARCH_STOPWORDS = {
    "about", "after", "and", "before", "for", "from", "how", "into",
    "latest", "of", "on", "or", "the", "to", "what", "when", "which",
    "who", "why", "with",
}


class SearchResults(list[SearchResult]):
    """Keep per-call diagnostics even when all branches return no matches."""

    def __init__(self, results: list[SearchResult], diagnostics: dict):
        super().__init__(results)
        self.diagnostics = diagnostics


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
        selected = rank_search_results(deduplicated, search_query.query)[:max_results]
        successful_branches = [
            name for name, _, error in outcomes if error is None
        ]
        diagnostics = {
            "provider_counts": {
                provider: sum(result.provider.value == provider for result in results)
                for provider in dict.fromkeys([
                    *("tavily" if name == "web" else name for name, _, _ in outcomes),
                    *(result.provider.value for result in results),
                ])
            },
            "successful_branches": successful_branches,
            "failed_branches": failures,
            "web_fallback_activated": any(
                result.provider == SourceProvider.BRAVE for result in results
            ),
        }
        # Attach only safe branch-level diagnostics so the graph can report
        # partial progress without retaining errors or provider responses.
        selected = [
            result.model_copy(update={
                "provider_metadata": {
                    **result.provider_metadata,
                    "search_diagnostics": diagnostics,
                },
            })
            for result in selected
        ]
        print(
            "\nParallel search completed:",
            f"query id={search_query.id}",
            f"query={search_query.query}",
            f"results={len(selected)}",
            f"discovered_provider_counts={diagnostics['provider_counts']}",
            f"failed_branches={failures}",
        )
        return SearchResults(selected, diagnostics)

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
    """
    Merge results that identify the same page, paper, or article.
    Keep one result per identifiable source, while remembering every provider result that led us to it.
    """

    groups: list[tuple[set[str], list[SearchResult]]] = []
    for result in results:
        identities = _identity_keys(result)
        # Find every existing group that shares at least one known identifier with this new result
        matches = [group for group in groups if group[0] & identities]
        if not matches:
            # We have not seen anything identifying this source yet. Start a collection for it.
            groups.append((identities, [result]))
            continue

        target = matches[0]
        # adds the new identifiers to the existing set. 
        # Sets automatically avoid duplicate entries.
        target[0].update(identities)
        target[1].append(result)

        # DOI, arXiv ID, page ID, and URL aliases can reveal that two earlier
        # groups describe the same underlying source.
        for duplicate in matches[1:]:
            target[0].update(duplicate[0])
            target[1].extend(duplicate[1])
            groups.remove(duplicate)

    return [_merge_group(group_results) for _, group_results in groups]


def rank_search_results(
    results: list[SearchResult], query: str, *, provider_offset: int | None = None,
) -> list[SearchResult]:
    """Interleave provider buckets ranked by the same query-fit signals."""
    tokens = _search_tokens(query)
    # Diversity is a tie-breaking opportunity, not an obligation to promote a
    # provider's zero-overlap result. Keep those as fallback, not forced slots.
    eligible = [result for result in results if not tokens or tokens & _search_tokens(
        f"{result.title} {result.snippet or ''}"
    )]
    deferred = [result for result in results if result not in eligible]
    if not eligible:
        eligible, deferred = results, []
    buckets: dict[SourceProvider, list[SearchResult]] = {}
    for result in sorted(eligible, key=lambda item: _result_rank(item, query), reverse=True):
        buckets.setdefault(result.provider, []).append(result)
    # Extraction rotates a stable provider cycle, independent of which provider
    # has the highest text score on each individual query.
    groups = (list(buckets.values()) if provider_offset is None
              else [buckets[provider] for provider in SourceProvider if provider in buckets])
    if not groups:
        return []
    # Rotate the first provider across extraction intents so a small page budget
    # does not always spend each query's first opportunity on the same provider.
    offset = (provider_offset or 0) % len(groups)
    groups = groups[offset:] + groups[:offset]
    return [item for row in zip_longest(*groups) for item in row if item is not None] + deferred


def _result_rank(result: SearchResult, query: str) -> tuple[float, int]:
    """Use shared text/date signals, never incomparable raw provider scores."""
    # break query in tokens
    query_tokens = _search_tokens(query)
    # Calculate how much of the query appears in the title
    title_overlap = _overlap(query_tokens, _search_tokens(result.title))
    # Calculate the same overlap for the snippet
    snippet_overlap = _overlap(query_tokens, _search_tokens(result.snippet or ""))
    provenance = result.provider_metadata.get("discovery_provenance", [])
    # Give a small preference to a source encountered through multiple discoveries.
    provenance_count = len(provenance)
    # check for year match if present in query
    query_years = set(_YEAR.findall(query))
    source_dates = " ".join(str(result.provider_metadata.get(key) or "") for key in (
        "published_at", "updated_at", "last_edited_at",
    ))
    temporal_match = 0.3 if query_years & set(_YEAR.findall(source_dates)) else 0.0
    combined = (
        (3.0 * title_overlap)
        + (1.5 * snippet_overlap)
        + (0.25 * min(max(provenance_count - 1, 0), 2))
        + temporal_match
    )
    return combined, provenance_count


def _search_tokens(value: str) -> set[str]:
    # Removes words in your configured stopword list
    # Returns a set, so repeated words appear only once.
    return {
        token for token in _TOKEN.findall(value.casefold())
        if token not in _SEARCH_STOPWORDS
    }


def _overlap(expected: set[str], actual: set[str]) -> float:
    if not expected:
        return 0.0
    return len(expected & actual) / len(expected)


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
