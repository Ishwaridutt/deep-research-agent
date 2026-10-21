"""Focused offline tests for deterministic public-source cleaning."""

import asyncio

import pytest

from src.agents.research.research_source_processor_node import (
    research_source_processor_node,
)
from src.graph.graph_state import create_initial_workflow_state
from src.models import ResearchSource, SearchQuery, SearchResult, SourceProvider
from src.tools.cleaning import SourceCleaner, SourceCleaningError
from src.tools.extraction import ExtractedSourceContent


def test_cleaner_preserves_structure_and_removes_html_noise():
    cleaner = SourceCleaner(min_chars=20, max_chars=1_000)
    cleaned = cleaner.clean(
        """
        <html><body>
        <nav>Site navigation</nav>
        <article>
          <h1>ＡＩ Research</h1>
          <p>Agentic   AI supports inter-\nnational teams.</p>
          <ul><li>Finding one</li><li>Finding two</li></ul>
        </article>
        <script>privateDebugValue()</script>
        </body></html>
        """,
        content_format="html",
    )

    assert "# AI Research" in cleaned.content
    assert "international teams" in cleaned.content
    assert "- Finding one" in cleaned.content
    assert "Site navigation" not in cleaned.content
    assert "privateDebugValue" not in cleaned.content
    assert cleaned.version == "deterministic-v1"


def test_cleaner_enforces_quality_and_size_limits():
    cleaner = SourceCleaner(min_chars=10, max_chars=24)

    with pytest.raises(SourceCleaningError):
        cleaner.clean("Too short")

    cleaned = cleaner.clean("A sufficiently long source paragraph for truncation.")
    assert cleaned.truncated is True
    assert cleaned.cleaned_chars <= 24


def test_source_processor_cleans_content_and_records_provenance():
    class FakeExtractor:
        async def extract(self, *, search_result: SearchResult) -> ExtractedSourceContent:
            return ExtractedSourceContent(
                url=search_result.url,
                content="<article><h1>Research</h1><p>Useful   evidence for the report.</p></article>",
                content_format="html",
                provider_metadata={"extraction_provider": "firecrawl"},
            )

    async def run() -> None:
        state = create_initial_workflow_state(
            job_id="job-1",
            research_session_id="session-1",
            question="What does the source say?",
            allowed_sources=[ResearchSource.WEB],
        )
        state["research"]["search_queries"] = [SearchQuery(
            id="query-1",
            subquestion_id="subquestion-1",
            query="research evidence",
            purpose="Find evidence",
            iteration=1,
        )]
        state["research"]["search_results"] = [SearchResult(
            id="result-1",
            query_id="query-1",
            provider=SourceProvider.TAVILY,
            canonical_id="web:https://example.com/research",
            title="Research",
            url="https://example.com/research",
        )]

        result = await research_source_processor_node(
            state,
            source_extractor=FakeExtractor(),
            source_cleaner=SourceCleaner(min_chars=20, max_chars=1_000),
        )
        source = result["research"]["sources"][0]

        assert source.content == "# Research\n\nUseful evidence for the report."
        assert source.provider_metadata["cleaner_version"] == "deterministic-v1"
        assert source.provider_metadata["content_truncated"] is False
        assert source.provider_metadata["extraction_provider"] == "firecrawl"

    asyncio.run(run())


def test_source_processor_reserves_extraction_capacity_per_round():
    class FakeExtractor:
        calls = 0

        async def extract(self, *, search_result: SearchResult) -> ExtractedSourceContent:
            self.calls += 1
            return ExtractedSourceContent(
                url=search_result.url,
                content="Detailed evidence content for the research report.",
            )

    async def run():
        state = create_initial_workflow_state(
            job_id="job-1",
            research_session_id="session-1",
            question="What does the evidence show?",
            allowed_sources=[ResearchSource.WEB],
        )
        state["research_control"].budget.max_extracted_pages_per_round = 20
        state["research"]["search_queries"] = [SearchQuery(
            id="query-1",
            subquestion_id="subquestion-1",
            query="research evidence",
            purpose="Find evidence",
            iteration=1,
        )]
        state["research"]["search_results"] = [
            SearchResult(
                id=f"result-{index}",
                query_id="query-1",
                provider=SourceProvider.TAVILY,
                canonical_id=f"web:https://example.com/{index}",
                title=f"Research evidence {index}",
                url=f"https://example.com/{index}",
            )
            for index in range(25)
        ]
        extractor = FakeExtractor()
        update = await research_source_processor_node(
            state,
            source_extractor=extractor,
            source_cleaner=SourceCleaner(min_chars=20, max_chars=1_000),
        )
        return extractor.calls, update

    calls, update = asyncio.run(run())

    assert calls == 20
    assert len(update["research"]["sources"]) == 20
    assert update["research_control"].usage.page_extraction_attempts == 20


def test_extraction_cap_balances_queries_within_and_across_subquestions():
    from src.models import SubQuestion

    class Extractor:
        def __init__(self):
            self.calls = []

        async def extract(self, *, search_result):
            self.calls.append(search_result.query_id)
            return ExtractedSourceContent(url=search_result.url, content="Specific evidence supporting a distinct research intent.")

    async def run():
        state = create_initial_workflow_state(job_id="job", research_session_id="session",
                                              question="Explain the topic", allowed_sources=[ResearchSource.WEB])
        state["research_control"].budget = state["research_control"].budget.model_copy(update={"max_extracted_pages_per_round": 4})
        state["research"]["subquestions"] = [SubQuestion(id=task, question=task, objective="Research") for task in ("a", "b")]
        state["research"]["search_queries"] = [SearchQuery(
            id=f"{task}-{intent}", subquestion_id=task, query=f"topic {intent}", purpose="Find evidence", iteration=1,
        ) for task in ("a", "b") for intent in (1, 2)]
        state["research"]["search_results"] = [SearchResult(
            id=f"{query.id}-{index}", query_id=query.id, provider=SourceProvider.TAVILY,
            canonical_id=f"web:{query.id}/{index}", title=query.query,
            url=f"https://example.com/{query.id}/{index}", provider_score=1.0 if query.id.endswith("1") else 0.1,
        ) for query in state["research"]["search_queries"] for index in range(4)]
        extractor = Extractor()
        update = await research_source_processor_node(state, source_extractor=extractor, source_cleaner=SourceCleaner(min_chars=20, max_chars=1000))
        assert extractor.calls == ["a-1", "b-1", "a-2", "b-2"]
        stats = update["research"]["extraction_stats"]
        assert stats == {"iteration": 1, "extracted_provider_counts": {"tavily": 4}, "extracted_count": 4,
                         "reused_count": 0, "attempted_count": 4, "deferred_count": 12, "carried_over_count": 0}
        assert len(update["research"]["deferred_search_result_ids"]) == 12
        # Previously extracted pages remain reusable without another API call.
        state.update(update)
        again = await research_source_processor_node(state, source_extractor=extractor, source_cleaner=SourceCleaner(min_chars=20, max_chars=1000))
        assert again["research"]["extraction_stats"]["reused_count"] == 4
        assert again["research"]["extraction_stats"]["extracted_count"] == 4
        assert len(extractor.calls) == 8

    asyncio.run(run())
