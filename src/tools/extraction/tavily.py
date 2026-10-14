"""Tavily implementation of the WebExtractor contract."""

from typing import Literal

from tavily import AsyncTavilyClient

from src.tools.extraction.base import (
    ExtractedWebPage,
    WebExtractionError,
    WebExtractor,
)
from src.tools.provider_resilience import fallback_allowed_for_exception


class TavilyWebExtractor(WebExtractor):
    """Extracts webpage content using Tavily Extract."""

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: int,
        extract_depth: Literal[
            "basic",
            "advanced",
        ],
        output_format: Literal[
            "markdown",
            "text",
        ],
    ) -> None:
        """Creates the Tavily extraction adapter."""

        self._client = AsyncTavilyClient(api_key=api_key,)

        self._timeout_seconds = timeout_seconds
        self._extract_depth = extract_depth
        self._output_format = output_format

    async def extract(
        self,
        *,
        url: str,
    ) -> ExtractedWebPage:
        """Extracts one URL through Tavily."""

        try:
            response = await self._client.extract(
                urls=[url],
                include_images=False,
                extract_depth=self._extract_depth,
                format=self._output_format,
                timeout=self._timeout_seconds,
            )

        except Exception as exc:
            print("Tavily extraction failed:", type(exc).__name__)
            raise WebExtractionError(
                f"Tavily extraction failed for '{url}'.",
                fallback_allowed=fallback_allowed_for_exception(exc),
            ) from exc

        results = response.get("results", [])

        if not results:
            raise WebExtractionError(
                f"Tavily could not extract '{url}'."
            )

        result = results[0]

        extracted_url = str(result.get("url") or url).strip()
        content = str(result.get("raw_content") or "").strip()

        if not content:
            raise WebExtractionError(
                (
                    "Tavily returned no usable content "
                    f"for '{url}'."
                )
            )

        print("Extracted data for URL:", extracted_url)
        return ExtractedWebPage(
            url=extracted_url,
            content=content,
            content_format=self._output_format,
            provider_metadata={"extraction_provider": "tavily"},
        )

    async def close(
        self,
    ) -> None:
        """Closes Tavily's underlying async HTTP client."""

        await self._client.close()
