"""Focused tests for parallel public search and canonical deduplication."""

import asyncio

import pytest

from src.models import SearchQuery, SearchResult, SourceProvider
from src.tools.search.base import SearchProviderError
from src.tools.search.parallel_search import (
    ParallelSearchProvider,
    deduplicate_search_results,
    rank_search_results,
    _result_rank,
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
        "provider_counts": {"tavily": 0, "wikipedia": 1},
        "successful_branches": ["wikipedia"],
        "failed_branches": ["web"],
        "web_fallback_activated": False,
    }


def test_discovery_counts_survive_aggregate_ranking_and_empty_results():
    class Provider:
        def __init__(self, results):
            self.results = results

        async def search(self, **kwargs):
            return self.results

    composite = ParallelSearchProvider(branches=[
        ("web", Provider([_result(result_id="web", provider=SourceProvider.TAVILY,
                                  canonical_id="web:1", url="https://example.com/1", provider_score=1.0)]), 5),
        ("arxiv", Provider([_result(result_id="paper", provider=SourceProvider.ARXIV,
                                    canonical_id="arxiv:1", url="https://arxiv.org/abs/2401.12345")]), 3),
        ("wikipedia", Provider([]), 3),
    ], max_concurrency=3)
    results = asyncio.run(composite.search(search_query=_query(), max_results=1))
    assert len(results) == 1 and results[0].provider == SourceProvider.TAVILY
    assert results.diagnostics["provider_counts"] == {"tavily": 1, "arxiv": 1, "wikipedia": 0}
    empty = ParallelSearchProvider(branches=[("wikipedia", Provider([]), 3)], max_concurrency=1)
    results = asyncio.run(empty.search(search_query=_query(), max_results=5))
    assert results == [] and results.diagnostics["provider_counts"] == {"wikipedia": 0}


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


def test_search_node_preserves_discovery_counts_before_its_score_filter():
    from types import SimpleNamespace
    from src.agents.research.research_web_search_node import research_web_search_node
    from src.graph.graph_state import create_initial_workflow_state
    from src.models import ResearchSource
    from src.tools.search.parallel_search import SearchResults

    class Provider:
        async def search(self, **kwargs):
            return SearchResults([_result(result_id="kept", provider=SourceProvider.TAVILY,
                                           canonical_id="web:1", url="https://example.com/1", provider_score=0.9)], {
                "provider_counts": {"tavily": 5, "arxiv": 3, "wikipedia": 0},
                "failed_branches": [], "web_fallback_activated": False,
            })

    state = create_initial_workflow_state(job_id="job", research_session_id="session",
                                          question="Research the topic", allowed_sources=[ResearchSource.WEB])
    state["research"]["search_queries"] = [_query()]
    settings = SimpleNamespace(research_limits=SimpleNamespace(max_total_queries_per_job=30, max_search_results_per_query=5),
                               search=SimpleNamespace(min_provider_score=0.7))
    update = asyncio.run(research_web_search_node(state, search_provider=Provider(), settings=settings))
    assert len(update["research"]["search_results"]) == 1
    assert update["research"]["search_stats"]["provider_counts"] == {"tavily": 5, "arxiv": 3, "wikipedia": 0}
    assert update["research"]["search_stats"]["iteration"] == 1


def test_shared_ranking_does_not_penalize_missing_provider_scores():
    scored = _result(result_id="web", provider=SourceProvider.TAVILY, canonical_id="web:1",
                     url="https://example.com/1", title="Agentic AI", provider_score=0.99,
                     metadata={"discovery_provenance": [{"provider_score": 0.99}]})
    native = _result(result_id="native", provider=SourceProvider.ARXIV, canonical_id="arxiv:1",
                     url="https://arxiv.org/abs/2401.12345", title="Agentic AI",
                     metadata={"discovery_provenance": [{"provider_score": None}]})
    assert _result_rank(scored, _query().query) == _result_rank(native, _query().query)


def test_provider_interleaving_respects_limits_fills_empty_buckets_and_rotates():
    results = [_result(result_id=f"web-{n}", provider=SourceProvider.TAVILY,
                       canonical_id=f"web:{n}", url=f"https://example.com/{n}", title="Agentic AI", provider_score=0.99)
               for n in range(5)]
    results += [_result(result_id=provider.value, provider=provider, canonical_id=f"{provider.value}:1",
                        url=f"https://example.com/{provider.value}", title="Agentic AI")
                for provider in (SourceProvider.ARXIV, SourceProvider.WIKIPEDIA)]
    ranked = rank_search_results(results, _query().query)
    assert [item.provider for item in ranked[:3]] == [SourceProvider.TAVILY, SourceProvider.ARXIV, SourceProvider.WIKIPEDIA]
    assert len(ranked) == 7 and len({item.id for item in ranked}) == 7
    assert [item.id for item in ranked[3:]] == [f"web-{n}" for n in range(1, 5)]
    assert rank_search_results(results, _query().query, provider_offset=1)[0].provider == SourceProvider.ARXIV
    assert rank_search_results([], _query().query) == []
    for index, best_match in enumerate((SourceProvider.TAVILY, SourceProvider.WIKIPEDIA, SourceProvider.ARXIV)):
        varied = [item.model_copy(update={"title": "Agentic AI" if item.provider == best_match else "Other topic"}) for item in results]
        assert rank_search_results(varied, _query().query, provider_offset=index)[0].provider == best_match


def test_parallel_selection_keeps_native_candidates_despite_high_web_scores():
    class Provider:
        def __init__(self, provider):
            self.provider = provider

        async def search(self, *, search_query, max_results):
            return [_result(result_id=f"{self.provider.value}-{n}", provider=self.provider,
                            canonical_id=f"{self.provider.value}:{n}", url=f"https://example.com/{self.provider.value}/{n}",
                            title="Agentic AI", provider_score=0.99 if self.provider == SourceProvider.TAVILY else None)
                    for n in range(max_results)]

    composite = ParallelSearchProvider(branches=[
        ("web", Provider(SourceProvider.TAVILY), 5), ("arxiv", Provider(SourceProvider.ARXIV), 3),
        ("wikipedia", Provider(SourceProvider.WIKIPEDIA), 3),
    ], max_concurrency=3)
    selected = asyncio.run(composite.search(search_query=_query(), max_results=5))
    assert len(selected) == 5
    assert {item.provider for item in selected} == {SourceProvider.TAVILY, SourceProvider.ARXIV, SourceProvider.WIKIPEDIA}
