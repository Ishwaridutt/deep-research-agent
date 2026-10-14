"""Firecrawl implementation of the WebExtractor contract."""

import httpx

from src.tools.extraction.base import (
    ExtractedWebPage,
    WebExtractionError,
    WebExtractor,
)
from src.tools.provider_resilience import fallback_allowed_for_status


class FirecrawlWebExtractor(WebExtractor):
    """Extracts main-page Markdown through Firecrawl's scrape API."""

    def __init__(
        self,
        *,
        api_key: str,
        api_url: str,
        timeout_seconds: int,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_url = api_url
        self._timeout_seconds = timeout_seconds
        self._owns_client = client is None
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = client or httpx.AsyncClient()

    async def extract(self, *, url: str) -> ExtractedWebPage:
        try:
            response = await self._client.post(
                self._api_url,
                json={
                    "url": url,
                    "formats": ["markdown"],
                    "onlyMainContent": True,
                },
                headers=self._headers,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise WebExtractionError(
                f"Firecrawl extraction failed for '{url}'.",
                fallback_allowed=fallback_allowed_for_status(exc.response.status_code),
            ) from exc
        except (httpx.RequestError, ValueError) as exc:
            raise WebExtractionError(
                f"Firecrawl extraction failed for '{url}'."
            ) from exc

        if not isinstance(payload, dict):
            raise WebExtractionError(
                f"Firecrawl returned invalid data for '{url}'."
            )
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            raise WebExtractionError(
                f"Firecrawl returned invalid data for '{url}'."
            )
        content = str(data.get("markdown") or "").strip()
        if not payload.get("success") or not content:
            raise WebExtractionError(
                f"Firecrawl returned no usable content for '{url}'."
            )

        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        extracted_url = str(
            metadata.get("sourceURL")
            or metadata.get("url")
            or url
        ).strip()

        print("Extracted data with Firecrawl for URL:", extracted_url)
        return ExtractedWebPage(
            url=extracted_url,
            content=content,
            content_format="markdown",
            provider_metadata={
                "extraction_provider": "firecrawl",
                "extraction_status_code": metadata.get("statusCode"),
            },
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
