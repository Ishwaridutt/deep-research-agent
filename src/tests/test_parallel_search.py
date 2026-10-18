"""Focused tests for parallel public search and canonical deduplication."""

import asyncio

import pytest

from src.models import SearchQuery, SearchResult, SourceProvider
from src.tools.search.base import SearchProviderError
from src.tools.search.parallel_search import (
    ParallelSearchProvider,
    deduplicate_search_results,
)


def _query() -> SearchQuery:
    return SearchQuery(
        id="query-1",
        subquestion_id="subquestion-1",
        query="agentic AI",
        purpose="Find sources",
        iteration=1,
    )


def _result(
    *,
    result_id: str,
    provider: SourceProvider,
    canonical_id: str,
    url: str,
    metadata: dict | None = None,
    title: str | None = None,
    snippet: str | None = None,
    provider_score: float | None = None,
) -> SearchResult:
    return SearchResult(
        id=result_id,
        query_id="query-1",
        provider=provider,
        canonical_id=canonical_id,
        title=title or result_id,
        url=url,
        snippet=snippet,
        provider_score=provider_score,
        provider_metadata=metadata or {},
    )


def test_parallel_search_honors_concurrency_and_provider_limits():
    tracker = {"active": 0, "maximum": 0}

    class FakeProvider:
        def __init__(self, name: str):
            self.name = name
            self.max_results = 0

        async def search(self, *, search_query, max_results):
            self.max_results = max_results
            tracker["active"] += 1
            tracker["maximum"] = max(tracker["maximum"], tracker["active"])
            await asyncio.sleep(0.01)
            tracker["active"] -= 1
            return [_result(
                result_id=self.name,
                provider=SourceProvider.TAVILY,
                canonical_id=f"web:https://example.com/{self.name}",
                url=f"https://example.com/{self.name}",
            )]

        async def close(self):
            pass

    providers = [FakeProvider(name) for name in ("web", "arxiv", "wikipedia")]
    composite = ParallelSearchProvider(
        branches=[
            ("web", providers[0], 5),
            ("arxiv", providers[1], 3),
            ("wikipedia", providers[2], 2),
        ],
        max_concurrency=2,
    )

    results = asyncio.run(composite.search(search_query=_query(), max_results=4))

    assert len(results) == 3
    assert tracker["maximum"] == 2
    assert [provider.max_results for provider in providers] == [4, 3, 2]


def test_parallel_search_applies_one_ranked_aggregate_limit():
    class Provider:
        def __init__(self, results):
            self.results = results

        async def search(self, *, search_query, max_results):
            return self.results

        async def close(self):
            pass

    relevant = _result(
        result_id="relevant",
        provider=SourceProvider.WIKIPEDIA,
        canonical_id="wikipedia:1",
        url="https://en.wikipedia.org/wiki/Agentic_AI",
        title="Agentic AI systems",
        snippet="Agentic AI architectures and applications.",
    )
    composite = ParallelSearchProvider(
        branches=[
            ("web", Provider([
                _result(
                    result_id="web-1",
                    provider=SourceProvider.TAVILY,
                    canonical_id="web:https://example.com/1",
                    url="https://example.com/1",
                    provider_score=0.2,
                ),
                _result(
                    result_id="web-2",
                    provider=SourceProvider.TAVILY,
                    canonical_id="web:https://example.com/2",
                    url="https://example.com/2",
                    provider_score=0.1,
                ),
            ]), 5),
            ("wikipedia", Provider([relevant]), 3),
        ],
        max_concurrency=2,
    )

    results = asyncio.run(composite.search(search_query=_query(), max_results=2))

    assert len(results) == 2
    assert results[0].id == "relevant"


def test_canonical_dedup_prefers_native_results_and_merges_provenance():
    results = deduplicate_search_results([
        _result(
            result_id="web-arxiv",
            provider=SourceProvider.TAVILY,
            canonical_id="web:https://arxiv.org/abs/2401.12345v2",
            url="https://arxiv.org/abs/2401.12345v2",
        ),
        _result(
            result_id="native-arxiv",
            provider=SourceProvider.ARXIV,
            canonical_id="arxiv:2401.12345:v2",
            url="https://arxiv.org/abs/2401.12345v2",
            metadata={"arxiv_id": "2401.12345", "doi": "10.1000/paper"},
        ),
        _result(
            result_id="doi-copy",
            provider=SourceProvider.BRAVE,
            canonical_id="web:https://doi.org/10.1000/paper",
            url="https://doi.org/10.1000/paper",
        ),
        _result(
            result_id="web-wikipedia",
            provider=SourceProvider.TAVILY,
            canonical_id="web:https://en.wikipedia.org/wiki/Artificial_intelligence",
            url="https://en.wikipedia.org/wiki/Artificial_intelligence",
        ),
        _result(
            result_id="native-wikipedia",
            provider=SourceProvider.WIKIPEDIA,
            canonical_id="wikipedia:123",
            url="https://en.wikipedia.org/wiki/Artificial_intelligence",
            metadata={"page_id": 123},
        ),
        _result(
            result_id="web-page",
            provider=SourceProvider.TAVILY,
            canonical_id="web:https://example.com/Article",
            url="https://example.com/Article/",
        ),
        _result(
            result_id="web-page-copy",
            provider=SourceProvider.BRAVE,
            canonical_id="web:https://example.com/Article",
            url="https://example.com/Article#section",
        ),
    ])

    assert len(results) == 3
    assert results[0].provider == SourceProvider.ARXIV
    assert results[1].provider == SourceProvider.WIKIPEDIA
    assert results[2].provider == SourceProvider.TAVILY
    assert {
        item["provider"]
        for item in results[0].provider_metadata["discovery_provenance"]
    } == {"tavily", "arxiv", "brave"}


def test_parallel_search_preserves_successful_branch_results():
    class Provider:
        def __init__(self, *, fails: bool):
            self.fails = fails

        async def search(self, *, search_query, max_results):
            if self.fails:
                raise SearchProviderError("provider unavailable")
            return [_result(
                result_id="result-1",
                provider=SourceProvider.WIKIPEDIA,
                canonical_id="wikipedia:123",
                url="https://en.wikipedia.org/wiki/Artificial_intelligence",
                metadata={"page_id": 123},
            )]

        async def close(self):
            pass

    composite = ParallelSearchProvider(
        branches=[
            ("web", Provider(fails=True), 5),
            ("wikipedia", Provider(fails=False), 3),
        ],
        max_concurrency=2,
    )

    results = asyncio.run(composite.search(search_query=_query(), max_results=5))

    assert [result.provider for result in results] == [SourceProvider.WIKIPEDIA]
    assert results[0].provider_metadata["search_diagnostics"] == {
        "successful_branches": ["wikipedia"],
        "failed_branches": ["web"],
        "web_fallback_activated": False,
    }


def test_parallel_search_fails_only_when_every_branch_fails():
    class FailingProvider:
        async def search(self, *, search_query, max_results):
            raise SearchProviderError("provider unavailable")

        async def close(self):
            pass

    composite = ParallelSearchProvider(
        branches=[
            ("web", FailingProvider(), 5),
            ("arxiv", FailingProvider(), 3),
        ],
        max_concurrency=2,
    )

    with pytest.raises(SearchProviderError, match="All public search branches failed"):
        asyncio.run(composite.search(search_query=_query(), max_results=5))
