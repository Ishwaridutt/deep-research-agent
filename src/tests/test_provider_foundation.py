"""Focused offline tests for normalized provider provenance."""

import asyncio
import json
import logging

import pytest
from langchain_core.exceptions import OutputParserException

from src.config.logging import JsonFormatter
from src.llm import LLMCallMetadata, LLMRole, LLMService, LLMServiceError
from src.llm.service import _safe_provider_error_details
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


def test_provider_diagnostics_exclude_generated_output():
    class ProviderError(RuntimeError):
        status_code = 400
        body = {
            "error": {
                "code": "tool_use_failed",
                "type": "invalid_request_error",
                "failed_generation": "PRIVATE GENERATED CONTENT",
            }
        }

    details = _safe_provider_error_details(ProviderError())

    assert details == {
        "provider_exception_type": "ProviderError",
        "provider_status_code": 400,
        "provider_error_code": "tool_use_failed",
        "provider_error_category": "invalid_request_error",
    }
    assert "PRIVATE GENERATED CONTENT" not in str(details)


def test_json_logs_include_safe_provider_failure_diagnostics():
    record = logging.LogRecord(
        name="src.llm.service",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="LLM provider invocation failed.",
        args=(),
        exc_info=None,
    )
    record.event = "llm_provider_failed"
    record.provider = "groq"
    record.role = "evidence_extractor"
    record.schema = "EvidenceExtractionOutput"
    record.attempt = 1
    record.max_attempts = 2
    record.failure_type = "provider request failed"
    record.provider_exception_type = "BadRequestError"
    record.provider_status_code = 400
    record.provider_error_code = "invalid_request_error"
    record.document_count = 2
    record.web_result_count = 3

    output = json.loads(JsonFormatter().format(record))

    assert output["provider_status_code"] == 400
    assert output["provider_error_code"] == "invalid_request_error"
    assert output["schema"] == "EvidenceExtractionOutput"
    assert output["attempt"] == 1
    assert output["document_count"] == 2
    assert output["web_result_count"] == 3
