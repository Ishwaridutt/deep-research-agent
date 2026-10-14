"""arXiv paper content extractor."""

import asyncio
from pathlib import Path
import tempfile
from typing import Any

import httpx
from pypdf.errors import PyPdfError

from src.documents.document_parser import DocumentParser
from src.models import SearchResult, SourceProvider
from src.models.document_models import DocumentFormat, ParsedDocument
from src.tools.extraction.base import ExtractedSourceContent, SourceExtractionError


class ArxivContentExtractor:
    """Extracts bounded arXiv PDFs with an abstract-only fallback."""

    def __init__(
        self,
        *,
        parser: DocumentParser,
        timeout_seconds: int,
        max_pdf_bytes: int,
        max_pdf_pages: int,
        full_text_enabled: bool,
        user_agent: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._parser = parser
        self._timeout_seconds = timeout_seconds
        self._max_pdf_bytes = max_pdf_bytes
        self._max_pdf_pages = max_pdf_pages
        self._full_text_enabled = full_text_enabled
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            headers={"User-Agent": user_agent},
        )

    async def extract(
        self,
        *,
        search_result: SearchResult,
    ) -> ExtractedSourceContent:
        if search_result.provider != SourceProvider.ARXIV:
            raise SourceExtractionError("Expected an arXiv search result.")

        metadata = dict(search_result.provider_metadata)
        abstract_content = self._abstract_content(search_result)

        if not self._full_text_enabled:
            return self._abstract_result(search_result, metadata, abstract_content)

        pdf_url = str(metadata.get("pdf_url") or "").strip()
        if not pdf_url:
            return self._abstract_result(search_result, metadata, abstract_content)

        try:
            pdf = await self._download_pdf(pdf_url)
            parsed = await self._parse_pdf(search_result.canonical_id, pdf)
        except (
            httpx.HTTPError,
            OSError,
            PyPdfError,
            SourceExtractionError,
            ValueError,
        ):
            if not abstract_content:
                raise SourceExtractionError("arXiv paper extraction failed.")
            return self._abstract_result(search_result, metadata, abstract_content)

        content = self._full_text_content(search_result, parsed)
        metadata.update({
            "content_source": "pdf",
            "pdf_bytes": len(pdf),
            "pages_loaded": len(parsed.blocks),
        })

        print(
            "arXiv paper loaded:",
            f"canonical_id={search_result.canonical_id}",
            f"pages={len(parsed.blocks)}",
        )
        return ExtractedSourceContent(
            url=search_result.url,
            content=content,
            content_format="markdown",
            provider_metadata=metadata,
        )

    async def _download_pdf(self, url: str) -> bytes:
        data = bytearray()
        async with self._client.stream(
            "GET",
            url,
            timeout=self._timeout_seconds,
        ) as response:
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > self._max_pdf_bytes:
                raise SourceExtractionError("arXiv PDF exceeds the byte limit.")

            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                if len(data) > self._max_pdf_bytes:
                    raise SourceExtractionError("arXiv PDF exceeds the byte limit.")

        if not data.startswith(b"%PDF-"):
            raise SourceExtractionError("arXiv returned an invalid PDF.")
        return bytes(data)

    async def _parse_pdf(
        self,
        document_id: str,
        pdf: bytes,
    ) -> ParsedDocument:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as file:
            file.write(pdf)
            path = Path(file.name)

        try:
            return await asyncio.to_thread(
                self._parser.parse,
                document_id,
                path,
                DocumentFormat.PDF,
                self._max_pdf_pages,
            )
        finally:
            path.unlink(missing_ok=True)

    @staticmethod
    def _abstract_content(search_result: SearchResult) -> str:
        metadata = search_result.provider_metadata
        abstract = str(metadata.get("abstract") or "").strip()
        authors = metadata.get("authors") or []
        author_line = ", ".join(str(author) for author in authors if author)

        parts = [f"# {search_result.title}"]
        if author_line:
            parts.append(f"Authors: {author_line}")
        if abstract:
            parts.extend(["## Abstract", abstract])
        return "\n\n".join(parts) if abstract else ""

    @staticmethod
    def _full_text_content(
        search_result: SearchResult,
        parsed: ParsedDocument,
    ) -> str:
        pages = [
            f"## Page {block.location.page_start}\n\n{block.text}"
            for block in parsed.blocks
        ]
        return "\n\n".join([f"# {search_result.title}", *pages])

    @staticmethod
    def _abstract_result(
        search_result: SearchResult,
        metadata: dict[str, Any],
        content: str,
    ) -> ExtractedSourceContent:
        if not content:
            raise SourceExtractionError("arXiv paper contains no usable abstract.")

        metadata["content_source"] = "abstract"
        print(
            "arXiv abstract loaded:",
            f"canonical_id={search_result.canonical_id}",
        )
        return ExtractedSourceContent(
            url=search_result.url,
            content=content,
            content_format="markdown",
            provider_metadata=metadata,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
