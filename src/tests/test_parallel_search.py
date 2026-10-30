"""Focused tests for parallel public search and canonical deduplication."""

import asyncio

import pytest

from src.models import SearchQuery, SearchResult, SourceProvider
from src.tools.search.base import SearchProviderError
from src.tools.search.parallel_search import (
    ParallelSearchProvider,
    deduplicate_search_results,
)


def _query(providers=("web", "arxiv", "wikipedia")) -> SearchQuery:
    return SearchQuery(
        id="query-1",
        subquestion_id="subquestion-1",
        query="agentic AI",
        purpose="Find sources",
        iteration=1,
        providers=list(providers),
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


def test_native_branches_run_only_when_the_query_requests_them():
    calls = []

    class Provider:
        def __init__(self, name, provider):
            self.name, self.provider = name, provider

        async def search(self, *, search_query, max_results):
            calls.append(self.name)
            return [_result(result_id=self.name, provider=self.provider, canonical_id=f"{self.name}:1",
                            url=f"https://example.com/{self.name}")]

        async def close(self):
            pass

    composite = ParallelSearchProvider(branches=[
        ("web", Provider("web", SourceProvider.TAVILY), 5),
        ("arxiv", Provider("arxiv", SourceProvider.ARXIV), 3),
        ("wikipedia", Provider("wikipedia", SourceProvider.WIKIPEDIA), 3),
    ], max_concurrency=3)

    # Default queries use only general web search.
    default_query = SearchQuery(id="q", subquestion_id="s", query="muscle growth", purpose="Find", iteration=1)
    results = asyncio.run(composite.search(search_query=default_query, max_results=5))
    assert calls == ["web"] and [item.id for item in results] == ["web"]
    assert results.diagnostics["provider_counts"] == {"tavily": 1}

    # Web always runs; a requested native branch joins it without dropping results.
    calls.clear()
    results = asyncio.run(composite.search(search_query=_query(("arxiv",)), max_results=5))
    assert sorted(calls) == ["arxiv", "web"]
    assert {item.id for item in results} == {"web", "arxiv"}


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
    # Each branch is capped by max_results; the relevance gate ranks them later.
    assert [item.provider for item in results] == [SourceProvider.TAVILY, SourceProvider.ARXIV]
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
                               search=SimpleNamespace())
    update = asyncio.run(research_web_search_node(state, search_provider=Provider(), settings=settings))
    assert len(update["research"]["search_results"]) == 1
    assert update["research"]["search_stats"]["provider_counts"] == {"tavily": 5, "arxiv": 3, "wikipedia": 0}
    assert update["research"]["search_stats"]["iteration"] == 1


def test_search_node_runs_round_queries_concurrently_within_budget():
    from types import SimpleNamespace
    from src.agents.research.research_web_search_node import research_web_search_node
    from src.graph.graph_state import create_initial_workflow_state
    from src.models import ResearchSource

    tracker = {"active": 0, "max_active": 0, "queries": []}

    class Provider:
        async def search(self, *, search_query, max_results):
            tracker["queries"].append(search_query.id)
            tracker["active"] += 1
            tracker["max_active"] = max(tracker["max_active"], tracker["active"])
            await asyncio.sleep(0.01)
            tracker["active"] -= 1
            return [_result(result_id=search_query.id, provider=SourceProvider.TAVILY,
                            canonical_id=f"web:{search_query.id}", url=f"https://example.com/{search_query.id}")]

    state = create_initial_workflow_state(job_id="job", research_session_id="session",
                                          question="Research the topic", allowed_sources=[ResearchSource.WEB])
    state["research"]["search_queries"] = [
        _query().model_copy(update={"id": f"q-{n}"}) for n in range(4)
    ]
    settings = SimpleNamespace(research_limits=SimpleNamespace(max_total_queries_per_job=3, max_search_results_per_query=5),
                               search=SimpleNamespace())
    update = asyncio.run(research_web_search_node(state, search_provider=Provider(), settings=settings))
    # The job budget allows three queries; they run together and keep query order.
    assert tracker["max_active"] == 3
    assert [item.id for item in update["research"]["search_results"]] == ["q-0", "q-1", "q-2"]
    assert update["research_control"].usage.queries_executed == 3


def test_title_identity_key_normalizes_site_suffixes_and_skips_vague_titles():
    from src.tools.search.parallel_search import title_identity_key

    key = title_identity_key("International Society of Sports Nutrition Position Stand: protein and exercise.")
    assert key == title_identity_key("[PDF] International Society of Sports Nutrition position stand: protein and exercise - PMC")
    assert key == title_identity_key("International Society of Sports Nutrition position stand: protein and exercise | Springer Nature Link")
    assert title_identity_key("Home") is None
    assert title_identity_key("Weight Gain Recommendations for Athletes and Military ...") is None


def test_pubmed_and_pmc_copies_merge_and_prefer_full_text():
    pubmed = _result(result_id="pubmed", provider=SourceProvider.TAVILY, canonical_id="web:pubmed",
                     url="https://pubmed.ncbi.nlm.nih.gov/31247944",
                     title="Nutrition Recommendations for Bodybuilders in the Off-Season: A Narrative Review.")
    pmc = _result(result_id="pmc", provider=SourceProvider.TAVILY, canonical_id="web:pmc",
                  url="https://pmc.ncbi.nlm.nih.gov/articles/PMC6680710",
                  title="Nutrition Recommendations for Bodybuilders in the Off-Season: A Narrative Review - PMC")
    merged = deduplicate_search_results([pubmed, pmc])
    assert len(merged) == 1 and merged[0].id == "pmc"
    assert {item["result_id"] for item in merged[0].provider_metadata["discovery_provenance"]} == {"pubmed", "pmc"}


def test_search_node_records_provider_failures_and_still_raises_unexpected_errors():
    from types import SimpleNamespace
    from src.agents.research.research_web_search_node import research_web_search_node
    from src.graph.graph_state import create_initial_workflow_state
    from src.models import ResearchSource

    class Provider:
        def __init__(self, failure):
            self.failure = failure

        async def search(self, *, search_query, max_results):
            if search_query.id == "q-bad":
                raise self.failure
            return [_result(result_id=search_query.id, provider=SourceProvider.TAVILY,
                            canonical_id=f"web:{search_query.id}", url=f"https://example.com/{search_query.id}")]

    def state():
        value = create_initial_workflow_state(job_id="job", research_session_id="session",
                                              question="Research the topic", allowed_sources=[ResearchSource.WEB])
        value["research"]["search_queries"] = [_query().model_copy(update={"id": query_id}) for query_id in ("q-ok", "q-bad")]
        return value

    settings = SimpleNamespace(research_limits=SimpleNamespace(max_total_queries_per_job=30, max_search_results_per_query=5),
                               search=SimpleNamespace())
    update = asyncio.run(research_web_search_node(
        state(), search_provider=Provider(SearchProviderError("provider down")), settings=settings,
    ))
    # The failed query becomes a recoverable note; the other query's results are kept.
    assert [item.id for item in update["research"]["search_results"]] == ["q-ok"]
    assert update["research"]["search_stats"]["failed_query_count"] == 1
    assert update["errors"][0].recoverable and "provider down" in update["errors"][0].message

    with pytest.raises(ValueError, match="bug"):
        asyncio.run(research_web_search_node(state(), search_provider=Provider(ValueError("bug")), settings=settings))
