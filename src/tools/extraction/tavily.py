"""Tavily implementation of the WebExtractor contract."""

from typing import Literal

from tavily import AsyncTavilyClient

from src.tools.extraction.base import (
    ExtractedWebPage,
    WebExtractionError,
    WebExtractor,
)


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
            print('Error while extracting data:', exc)
            raise WebExtractionError(f"Tavily extraction failed for '{url}'.") from exc

        results = response.get("results", [])


        if not results:
            failed_results = response.get("failed_results", [])
            failure_message = self._get_failure_message(failed_results)

            raise WebExtractionError(
                (
                    f"Tavily could not extract '{url}'."
                    f"{failure_message}"
                )
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

        print('Extracted data for url---->')
        print(extracted_url)
        # print(content)
        return ExtractedWebPage(
            url=extracted_url,
            content=content,
        )

    async def close(
        self,
    ) -> None:
        """Closes Tavily's underlying async HTTP client."""

        await self._client.close()

    @staticmethod
    def _get_failure_message(
        failed_results: list,
    ) -> str:
        """Creates a short provider error message when available."""

        if not failed_results:
            return ""

        first_failure = failed_results[0]

        if not isinstance(first_failure, dict):
            return ""

        error_message = str(first_failure.get("error") or "").strip()

        if not error_message:
            return ""

        return f" Provider message: {error_message}"