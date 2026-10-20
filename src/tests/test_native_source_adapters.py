"""Offline tests for native arXiv and Wikipedia adapters."""

import asyncio
from pathlib import Path

import httpx

from src.models import SearchQuery, SearchResult, SourceProvider
from src.models.document_models import (
    DocumentSourceLocation,
    ParsedDocument,
    ParsedDocumentBlock,
)
from src.tools.extraction.arxiv import ArxivContentExtractor
from src.tools.extraction.wikipedia import WikipediaContentExtractor
from src.tools.search.arxiv_search import ArxivSearchProvider
from src.tools.search.wikipedia_search import WikipediaSearchProvider


def _query() -> SearchQuery:
    return SearchQuery(
        id="query-1",
        subquestion_id="subquestion-1",
        query="agentic AI",
        purpose="Find reliable sources",
        iteration=1,
    )


def test_arxiv_search_normalizes_atom_metadata():
    feed = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom"
          xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/2401.12345v2</id>
        <updated>2026-01-02T00:00:00Z</updated>
        <published>2026-01-01T00:00:00Z</published>
        <title>  Agentic   AI Systems  </title>
        <summary>A detailed abstract.</summary>
        <author><name>Ada Researcher</name></author>
        <category term="cs.AI" />
        <arxiv:primary_category term="cs.AI" />
        <arxiv:doi>10.1000/example</arxiv:doi>
        <link href="https://arxiv.org/pdf/2401.12345v2" type="application/pdf" />
      </entry>
    </feed>"""

    async def run():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=feed)
            )
        )
        provider = ArxivSearchProvider(
            api_url="https://export.arxiv.org/api/query",
            timeout_seconds=5,
            user_agent="test-agent",
            client=client,
        )
        try:
            return await provider.search(search_query=_query(), max_results=2)
        finally:
            await client.aclose()

    results = asyncio.run(run())

    assert len(results) == 1
    assert results[0].provider == SourceProvider.ARXIV
    assert results[0].canonical_id == "arxiv:2401.12345:v2"
    assert results[0].title == "Agentic AI Systems"
    assert results[0].provider_metadata["authors"] == ["Ada Researcher"]
    assert results[0].provider_metadata["doi"] == "10.1000/example"


def test_wikipedia_search_normalizes_page_identity_and_snippet():
    payload = {
        "query": {
            "search": [{
                "pageid": 123,
                "title": "Artificial intelligence",
                "snippet": '<span class="searchmatch">Artificial</span> intelligence',
                "wordcount": 4200,
                "timestamp": "2026-01-03T00:00:00Z",
            }]
        }
    }

    async def run():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=payload)
            )
        )
        provider = WikipediaSearchProvider(
            api_url="https://en.wikipedia.org/w/api.php",
            timeout_seconds=5,
            user_agent="test-agent",
            client=client,
        )
        try:
            return await provider.search(search_query=_query(), max_results=2)
        finally:
            await client.aclose()

    result = asyncio.run(run())[0]

    assert result.provider == SourceProvider.WIKIPEDIA
    assert result.canonical_id == "wikipedia:123"
    assert result.snippet == "Artificial intelligence"
    assert result.provider_metadata["page_id"] == 123


def test_wikipedia_extractor_retains_revision_provenance():
    result = SearchResult(
        id="result-1",
        query_id="query-1",
        provider=SourceProvider.WIKIPEDIA,
        canonical_id="wikipedia:123",
        title="Artificial intelligence",
        url="https://en.wikipedia.org/wiki/Artificial_intelligence",
        provider_metadata={"page_id": 123},
    )
    payload = {
        "query": {
            "redirects": [{"from": "AI", "to": "Artificial intelligence"}],
            "pages": [{
                "pageid": 123,
                "title": "Artificial intelligence",
                "canonicalurl": result.url,
                "extract": "Artificial intelligence is intelligence exhibited by machines.",
                "revisions": [{
                    "revid": 456,
                    "timestamp": "2026-01-04T00:00:00Z",
                }],
            }],
        }
    }

    async def run():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=payload)
            )
        )
        extractor = WikipediaContentExtractor(
            api_url="https://en.wikipedia.org/w/api.php",
            timeout_seconds=5,
            user_agent="test-agent",
            client=client,
        )
        try:
            return await extractor.extract(search_result=result)
        finally:
            await client.aclose()

    loaded = asyncio.run(run())

    assert loaded.content.startswith("Artificial intelligence")
    assert loaded.provider_metadata["revision_id"] == 456
    assert loaded.provider_metadata["revision_timestamp"] == "2026-01-04T00:00:00Z"


def test_arxiv_extractor_reuses_parser_with_pdf_page_limit():
    class FakeParser:
        def __init__(self):
            self.call: tuple[str, Path, object, int] | None = None

        def parse(self, document_id, path, document_format, max_pages):
            self.call = (document_id, path, document_format, max_pages)
            return ParsedDocument(
                document_id=document_id,
                total_pages=31,
                pages_processed=7,
                blocks=[ParsedDocumentBlock(
                    text="Full paper text",
                    location=DocumentSourceLocation(page_start=1, page_end=1),
                )],
            )

    result = SearchResult(
        id="result-1",
        query_id="query-1",
        provider=SourceProvider.ARXIV,
        canonical_id="arxiv:2401.12345:v2",
        title="Agentic AI Systems",
        url="https://arxiv.org/abs/2401.12345v2",
        provider_metadata={
            "abstract": "A detailed abstract.",
            "authors": ["Ada Researcher"],
            "pdf_url": "https://arxiv.org/pdf/2401.12345v2",
        },
    )
    parser = FakeParser()

    async def run():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, content=b"%PDF-fake")
            )
        )
        extractor = ArxivContentExtractor(
            parser=parser,
            timeout_seconds=5,
            max_pdf_bytes=100,
            max_pdf_pages=7,
            full_text_enabled=True,
            user_agent="test-agent",
            client=client,
        )
        try:
            return await extractor.extract(search_result=result)
        finally:
            await client.aclose()

    loaded = asyncio.run(run())

    assert parser.call is not None
    assert parser.call[0] == result.canonical_id
    assert parser.call[3] == 7
    assert loaded.provider_metadata["content_source"] == "pdf"
    assert loaded.url == result.url  # Citation stays on /abs/; content comes from the PDF.
    assert loaded.provider_metadata["pages_loaded"] == 1
    assert loaded.provider_metadata["pages_processed"] == 7
    assert loaded.provider_metadata["total_pages"] == 31
    assert loaded.provider_metadata["pdf_truncated"] is True
    assert "## Page 1" in loaded.content


def test_arxiv_extractor_falls_back_to_abstract_when_pdf_exceeds_limit():
    result = SearchResult(
        id="result-1",
        query_id="query-1",
        provider=SourceProvider.ARXIV,
        canonical_id="arxiv:2401.12345:v1",
        title="Agentic AI Systems",
        url="https://arxiv.org/abs/2401.12345v1",
        provider_metadata={
            "abstract": "A detailed abstract.",
            "pdf_url": "https://arxiv.org/pdf/2401.12345v1",
        },
    )

    async def run():
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={"content-length": "500"},
                    content=b"%PDF-fake",
                )
            )
        )
        extractor = ArxivContentExtractor(
            parser=object(),
            timeout_seconds=5,
            max_pdf_bytes=100,
            max_pdf_pages=5,
            full_text_enabled=True,
            user_agent="test-agent",
            client=client,
        )
        try:
            return await extractor.extract(search_result=result)
        finally:
            await client.aclose()

    loaded = asyncio.run(run())

    assert loaded.provider_metadata["content_source"] == "abstract"
    assert "A detailed abstract." in loaded.content
