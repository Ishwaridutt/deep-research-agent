"""Offline tests for Brave, Firecrawl, retries, and provider failover."""

import asyncio
import json

import httpx
import pytest

from src.models import SearchQuery, SearchResult, SourceProvider
from src.tools.extraction.base import ExtractedWebPage, WebExtractionError
from src.tools.extraction.fallback import FallbackWebExtractor
from src.tools.extraction.firecrawl import FirecrawlWebExtractor
from src.tools.search.base import SearchProviderError
from src.tools.search.brave_search import BraveSearchProvider
from src.tools.search.fallback_search import FallbackSearchProvider
from src.tools.provider_resilience import fallback_allowed_for_status


def _query() -> SearchQuery:
    return SearchQuery(
        id="query-1",
        subquestion_id="subquestion-1",
        query="agentic AI",
        purpose="Find sources",
        iteration=1,
    )


def _result() -> SearchResult:
    return SearchResult(
        id="result-1",
        query_id="query-1",
        provider=SourceProvider.BRAVE,
        canonical_id="web:https://example.com/article",
        title="Example",
        url="https://example.com/article",
    )


def test_brave_search_uses_official_contract_and_normalizes_results():
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Subscription-Token"] == "brave-key"
        assert request.headers["Api-Version"] == "2023-01-01"
        assert request.url.params["q"] == "agentic AI"
        assert request.url.params["count"] == "2"
        return httpx.Response(200, json={
            "web": {
                "results": [{
                    "title": "Example",
                    "url": "https://Example.com/Article/",
                    "description": "A useful result.",
                    "age": "1 day ago",
                    "language": "en",
                }]
            }
        })

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        provider = BraveSearchProvider(
            api_key="brave-key",
            api_url="https://api.search.brave.com/res/v1/web/search",
            api_version="2023-01-01",
            timeout_seconds=5,
            client=client,
        )
        try:
            return await provider.search(search_query=_query(), max_results=2)
        finally:
            await client.aclose()

    result = asyncio.run(run())[0]

    assert result.provider == SourceProvider.BRAVE
    assert result.canonical_id == "web:https://example.com/Article"
    assert result.snippet == "A useful result."


def test_firecrawl_extracts_markdown_and_records_provenance():
    def handle(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer firecrawl-key"
        assert json.loads(request.content) == {
            "url": "https://example.com/article",
            "formats": ["markdown"],
            "onlyMainContent": True,
        }
        return httpx.Response(200, json={
            "success": True,
            "data": {
                "markdown": "# Extracted article",
                "metadata": {
                    "sourceURL": "https://example.com/canonical",
                    "statusCode": 200,
                },
            },
        })

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        extractor = FirecrawlWebExtractor(
            api_key="firecrawl-key",
            api_url="https://api.firecrawl.dev/v2/scrape",
            timeout_seconds=5,
            client=client,
        )
        try:
            return await extractor.extract(url="https://example.com/article")
        finally:
            await client.aclose()

    page = asyncio.run(run())

    assert page.url == "https://example.com/canonical"
    assert page.content_format == "markdown"
    assert page.provider_metadata["extraction_provider"] == "firecrawl"


def test_search_retries_primary_then_uses_fallback_and_opens_circuit():
    class Primary:
        calls = 0

        async def search(self, *, search_query, max_results):
            self.calls += 1
            raise SearchProviderError("unavailable")

        async def close(self):
            pass

    class Fallback:
        calls = 0

        async def search(self, *, search_query, max_results):
            self.calls += 1
            return [_result()]

        async def close(self):
            pass

    primary = Primary()
    fallback = Fallback()
    provider = FallbackSearchProvider(
        primary=primary,
        fallback=fallback,
        max_retries=1,
        retry_delay_seconds=0,
        circuit_failure_threshold=1,
        circuit_recovery_seconds=60,
    )

    async def run():
        first = await provider.search(search_query=_query(), max_results=2)
        second = await provider.search(search_query=_query(), max_results=2)
        return first, second

    first, second = asyncio.run(run())

    assert first == second == [_result()]
    assert primary.calls == 2
    assert fallback.calls == 2


def test_non_retryable_search_error_does_not_activate_fallback():
    class Primary:
        async def search(self, *, search_query, max_results):
            raise SearchProviderError("invalid request", fallback_allowed=False)

        async def close(self):
            pass

    class Fallback:
        calls = 0

        async def search(self, *, search_query, max_results):
            self.calls += 1
            return [_result()]

        async def close(self):
            pass

    fallback = Fallback()
    provider = FallbackSearchProvider(
        primary=Primary(),
        fallback=fallback,
        max_retries=2,
        retry_delay_seconds=0,
        circuit_failure_threshold=1,
        circuit_recovery_seconds=60,
    )

    with pytest.raises(SearchProviderError, match="invalid request"):
        asyncio.run(provider.search(search_query=_query(), max_results=2))

    assert fallback.calls == 0


def test_unusable_primary_extraction_activates_firecrawl_fallback():
    class Primary:
        calls = 0

        async def extract(self, *, url):
            self.calls += 1
            raise WebExtractionError("empty content")

        async def close(self):
            pass

    class Fallback:
        calls = 0

        async def extract(self, *, url):
            self.calls += 1
            return ExtractedWebPage(
                url=url,
                content="# Recovered content",
                content_format="markdown",
                provider_metadata={"extraction_provider": "firecrawl"},
            )

        async def close(self):
            pass

    primary = Primary()
    fallback = Fallback()
    extractor = FallbackWebExtractor(
        primary=primary,
        fallback=fallback,
        max_retries=1,
        retry_delay_seconds=0,
        circuit_failure_threshold=2,
        circuit_recovery_seconds=60,
    )

    page = asyncio.run(extractor.extract(url="https://example.com/article"))

    assert page.provider_metadata["extraction_provider"] == "firecrawl"
    assert primary.calls == 2
    assert fallback.calls == 1


def test_http_failure_classification_matches_failover_policy():
    assert fallback_allowed_for_status(401)
    assert fallback_allowed_for_status(429)
    assert fallback_allowed_for_status(503)
    assert not fallback_allowed_for_status(400)
    assert not fallback_allowed_for_status(422)
