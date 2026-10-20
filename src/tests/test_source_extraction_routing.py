"""Focused tests for provider-aware extraction and retrieval provenance."""

import asyncio

from src.agents.research.research_retrieval_node import _web_documents
from src.agents.research.research_source_processor_node import (
    research_source_processor_node,
)
from src.graph.graph_state import create_initial_workflow_state
from src.models import ResearchSource, SearchQuery, SearchResult, SourceProvider
from src.tools.cleaning import SourceCleaner
from src.tools.extraction import (
    ExtractedSourceContent,
    ExtractedWebPage,
    ProviderAwareSourceExtractor,
)


def _result(provider: SourceProvider) -> SearchResult:
    return SearchResult(
        id="result-1",
        query_id="query-1",
        provider=provider,
        canonical_id=(
            "arxiv:2401.12345:v2"
            if provider == SourceProvider.ARXIV
            else "web:https://example.com/article"
        ),
        title="Agentic AI",
        url=(
            "https://arxiv.org/abs/2401.12345v2"
            if provider == SourceProvider.ARXIV
            else "https://example.com/article"
        ),
        provider_metadata={"discovery_provider": provider.value},
    )


def test_provider_aware_extractor_routes_native_and_web_results():
    calls: list[str] = []

    class Web:
        async def extract(self, *, url):
            calls.append(f"web:{url}")
            return ExtractedWebPage(url=url, content="web content")

        async def close(self):
            pass

    class Native:
        async def extract(self, *, search_result):
            calls.append(f"native:{search_result.canonical_id}")
            return ExtractedSourceContent(
                url=search_result.url,
                content="native content",
            )

        async def close(self):
            pass

    async def run():
        extractor = ProviderAwareSourceExtractor(
            web=Web(),
            arxiv=Native(),
            wikipedia=Native(),
        )
        native = await extractor.extract(search_result=_result(SourceProvider.ARXIV))
        web = await extractor.extract(search_result=_result(SourceProvider.BRAVE))
        return native, web

    native, web = asyncio.run(run())

    assert native.content == "native content"
    assert web.content == "web content"
    assert calls == [
        "native:arxiv:2401.12345:v2",
        "web:https://example.com/article",
    ]


def test_source_processor_preserves_native_provenance_for_retrieval():
    class Extractor:
        async def extract(self, *, search_result):
            return ExtractedSourceContent(
                url=search_result.url,
                content="Detailed arXiv evidence for agentic artificial intelligence.",
                content_format="text",
                provider_metadata={"content_source": "pdf", "pages_loaded": 4},
            )

    async def run():
        state = create_initial_workflow_state(
            job_id="job-1",
            research_session_id="session-1",
            question="What is agentic AI?",
            allowed_sources=[ResearchSource.WEB],
        )
        query = SearchQuery(
            id="query-1",
            subquestion_id="subquestion-1",
            query="agentic AI",
            purpose="Find papers",
            iteration=1,
        )
        state["research"]["search_queries"] = [query]
        state["research"]["search_results"] = [_result(SourceProvider.ARXIV)]

        update = await research_source_processor_node(
            state,
            source_extractor=Extractor(),
            source_cleaner=SourceCleaner(min_chars=20, max_chars=1_000),
        )
        updated_state = {**state, "research": update["research"]}
        return update["research"]["sources"][0], _web_documents(updated_state)[0]

    source, document = asyncio.run(run())

    assert source.provider == SourceProvider.ARXIV
    assert source.canonical_id == "arxiv:2401.12345:v2"
    assert source.provider_metadata["discovery_provider"] == "arxiv"
    assert source.provider_metadata["content_source"] == "pdf"
    assert document.metadata["provider"] == "arxiv"
    assert document.metadata["canonical_id"] == "arxiv:2401.12345:v2"
    assert document.metadata["provider_metadata"]["pages_loaded"] == 4


def test_shared_url_across_queries_keeps_native_extraction_and_all_discovery_ids():
    calls = []

    class Extractor:
        async def extract(self, *, search_result):
            calls.append(search_result)
            return ExtractedSourceContent(url=search_result.url, content="Detailed source evidence from the native paper extractor.")

    async def run():
        state = create_initial_workflow_state(job_id="job", research_session_id="session",
                                              question="What is agentic AI?", allowed_sources=[ResearchSource.WEB])
        native = _result(SourceProvider.ARXIV).model_copy(update={"id": "native", "query_id": "query-2"})
        web = native.model_copy(update={"id": "web", "query_id": "query-1", "provider": SourceProvider.TAVILY,
                                        "canonical_id": f"web:{native.url}", "provider_score": 0.99, "provider_metadata": {}})
        state["research"]["search_queries"] = [SearchQuery(id=f"query-{n}", subquestion_id="task", query="agentic AI", purpose="Research", iteration=1) for n in (1, 2)]
        state["research"]["search_results"] = [web, native]
        return await research_source_processor_node(state, source_extractor=Extractor(), source_cleaner=SourceCleaner(min_chars=20, max_chars=1000))

    update = asyncio.run(run())
    assert len(calls) == 1 and calls[0].provider == SourceProvider.ARXIV
    source = update["research"]["sources"][0]
    assert source.provider == SourceProvider.ARXIV
    assert set(source.search_result_ids) == {"web", "native"}
    assert update["research"]["extraction_stats"]["extracted_provider_counts"] == {"arxiv": 1}
