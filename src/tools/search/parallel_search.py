"""Parallel public search composition and canonical deduplication."""

import asyncio
import re
from urllib.parse import unquote, urlsplit

from src.guardrails import url_dedup_key
from src.models import SearchQuery, SearchResult, SourceProvider
from src.tools.search.base import SearchProvider, SearchProviderError


_ARXIV_VERSION = re.compile(r"v\d+$", re.IGNORECASE)
_TITLE_PREFIX = re.compile(r"^\s*(?:\[pdf\]\s*|title:\s*)+", re.IGNORECASE)
_TITLE_WORD = re.compile(r"[^\W_]+")
_MIN_TITLE_WORDS = 6
# Index pages that usually show only an abstract when a full-text copy exists elsewhere.
_ABSTRACT_ONLY_HOSTS = {"pubmed.ncbi.nlm.nih.gov"}


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
        # General web search always runs; arXiv and Wikipedia run only when the
        # query generator requested them for this query.
        # "web" is added again for safety, so web always runs
        requested = {"web", *getattr(search_query, "providers", ["web"])}
        # arXiv runs only if ARXIV_ENABLED is true and the query is tagged arxiv.
        branches = [branch for branch in self._branches if branch[0] in requested]
        skipped = sorted(requested - {name for name, _, _ in branches})
        if skipped:
            print(f"\nrequested search branches are not enabled (skipped): {skipped}")
        if not branches:
            return SearchResults(
                [],
                {
                    "provider_counts": {}, 
                    "successful_branches": [],
                    "failed_branches": [], 
                    "web_fallback_activated": False
                }
            )

        # Each branch keeps its own result cap; the semaphore limits total
        # outbound searches without forcing the providers to run sequentially.
        outcomes = await asyncio.gather(*(
            self._search_branch(
                name=name,
                provider=provider,
                search_query=search_query,
                max_results=min(limit, max_results),
            )
            for name, provider, limit in branches
        ))

        # A failed provider must not discard useful results returned by the others.
        # flatten every branch’s results into one list
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

        # Ranking happens later in the source processor's relevance gate, which
        # compares every candidate on one embedding scale regardless of provider.
        selected = deduplicate_search_results(results)
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
        # compute every identity key for the result (see _identity_keys)
        identities = _identity_keys(result)
        # Find every existing group that shares at least one known identifier key with this new result
        matches = [group for group in groups if group[0] & identities]
        if not matches:
            # We have not seen anything identifying this source yet. Start a collection for it.
            groups.append((identities, [result]))
            continue

        target = matches[0]
        # adds the new identifiers to the existing set, to the first matching group.
        # Sets automatically avoid duplicate entries.
        target[0].update(identities)
        target[1].append(result)

        # DOI, arXiv ID, page ID, and URL aliases can reveal that two earlier
        # groups describe the same underlying source.
        # if it matched several groups, those groups were the same source all along
        # The extra groups are merged into the first and removed.
        for duplicate in matches[1:]:
            target[0].update(duplicate[0])
            target[1].extend(duplicate[1])
            groups.remove(duplicate)
    # reduce each group to one result with _merge_group
    return [_merge_group(group_results) for _, group_results in groups]


def _merge_group(results: list[SearchResult]) -> SearchResult:
    native_providers = {SourceProvider.ARXIV, SourceProvider.WIKIPEDIA}

    # Native results carry the IDs and metadata required by their native extractors.
    representative = next(
        (result for result in results if result.provider in native_providers),
        preferred_full_text_result(results),
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


def preferred_full_text_result(results: list[SearchResult]) -> SearchResult:
    """Prefer a full-text copy over an abstract-only index page of the same work."""
    return next(
        (result for result in results if urlsplit(result.url).hostname not in _ABSTRACT_ONLY_HOSTS),
        results[0],
    )


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

    # The same article often appears on several hosts (PubMed abstract, PMC full
    # text, publisher page) with no shared URL or DOI in the search result.
    title_key = title_identity_key(result.title)
    if title_key:
        keys.add(title_key)

    try:
        keys.add(f"url:{url_dedup_key(result.url)}")
    except ValueError:
        pass
    return keys


def title_identity_key(title: str) -> str | None:
    """
    Normalize a specific title into a cross-host identity key.

    Site suffixes ("- PMC", "| Springer Nature Link") and prefixes ("[PDF]",
    "Title:") are removed. Short or truncated titles return None because they
    are not specific enough to prove two pages are the same work.
    """
    text = title.strip()
    if not text or "..." in text or "\u2026" in text:
        return None
    text = _TITLE_PREFIX.sub("", text)
    text = text.split(" | ")[0]
    if " - " in text:
        head = text.rsplit(" - ", 1)[0]
        if len(_TITLE_WORD.findall(head)) >= _MIN_TITLE_WORDS:
            text = head
    words = _TITLE_WORD.findall(text.casefold())
    if len(words) < _MIN_TITLE_WORDS:
        return None
    return "title:" + " ".join(words)


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
