"""Focused offline tests for normalized provider provenance."""

import asyncio

import pytest
from langchain_core.exceptions import OutputParserException

from src.llm import LLMCallMetadata, LLMRole, LLMService, LLMServiceError
from src.models import EvidenceExtractionOutput, SearchQuery, SourceProvider
from src.tools.search.tavily_search import TavilySearchProvider


def test_tavily_results_include_provider_and_canonical_identity():
    provider = object.__new__(TavilySearchProvider)
    query = SearchQuery(
        id="query-1",
        subquestion_id="subquestion-1",
        query="agentic AI",
        purpose="Find sources",
        iteration=1,
    )

    results = provider._normalize_results(
        search_query=query,
        raw_results=[{
            "title": "Example",
            "url": "HTTPS://Example.com/Article/",
            "content": "Snippet",
            "score": 0.8,
        }],
        max_results=5,
    )

    assert len(results) == 1
    assert results[0].provider == SourceProvider.TAVILY
    assert results[0].canonical_id == "web:https://example.com/Article"


def test_llm_errors_do_not_expose_provider_output():
    class FailingStructuredModel:
        async def ainvoke(self, messages, config):
            raise OutputParserException("private provider output")

    class FailingModel:
        def with_structured_output(self, output_schema, method):
            return FailingStructuredModel()

    service = LLMService(
        {LLMRole.EVIDENCE_EXTRACTOR: FailingModel()},
        structured_output_attempts=1,
    )

    with pytest.raises(LLMServiceError) as error:
        asyncio.run(service.invoke_structured(
            role=LLMRole.EVIDENCE_EXTRACTOR,
            messages=[],
            output_schema=EvidenceExtractionOutput,
            metadata=LLMCallMetadata(
                job_id="job-1",
                research_session_id="session-1",
                node="research_evidence_extractor",
            ),
        ))

    assert "private provider output" not in str(error.value)
    assert "invalid structured output" in str(error.value)
