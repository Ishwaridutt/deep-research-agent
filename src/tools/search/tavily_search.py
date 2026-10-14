"""Tavily implementation of the SearchProvider contract."""

from typing import Any

from tavily import AsyncTavilyClient

from src.guardrails import url_dedup_key
from src.models import SearchQuery, SearchResult, SourceProvider
from src.services import generate_search_result_id
from src.tools.provider_resilience import fallback_allowed_for_exception
from src.tools.search.base import (
    SearchProvider,
    SearchProviderError,
)


class TavilySearchProvider(SearchProvider):
    """Performs web searches using Tavily."""

    def __init__(
        self,
        *,
        api_key: str,
        timeout_seconds: int,
    ) -> None:
        """
        Creates a Tavily search provider.

        Tavily-specific configuration remains inside this adapter so the
        rest of the application stays provider-independent.
        """

        self._client = AsyncTavilyClient(api_key=api_key)
        self._timeout_seconds = timeout_seconds


    async def search(
        self,
        *,
        search_query: SearchQuery,
        max_results: int,
    ) -> list[SearchResult]:
        """
        Executes one Tavily search and normalizes its results.

        Search snippets are used only for source discovery. Full webpage
        content is retrieved later through the WebExtractor abstraction.
        """

        try:
            response = await self._client.search(
                query=search_query.query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=False,
                include_raw_content=False,
                include_images=False,
                timeout=self._timeout_seconds,
            )

        except Exception as exc:
            raise SearchProviderError(
                (
                    "Tavily search failed for query "
                    f"'{search_query.id}'."
                ),
                fallback_allowed=fallback_allowed_for_exception(exc),
            ) from exc

        raw_results = response.get(
            "results",
            [],
        )

        return self._normalize_results(
            search_query=search_query,
            raw_results=raw_results,
            max_results=max_results,
        )

    def _normalize_results(
        self,
        *,
        search_query: SearchQuery,
        raw_results: list[dict[str, Any]],
        max_results: int,
    ) -> list[SearchResult]:
        """
        Converts Tavily results into application-owned SearchResult models.

        Duplicate URLs and malformed results are discarded here so later
        pipeline stages receive a predictable result set.
        """

        normalized_results: list[SearchResult] = []

        seen_urls: set[str] = set()

        for raw_result in raw_results:
            if len(normalized_results) >= max_results:
                break

            title = str(
                raw_result.get("title") or ""
            ).strip()

            url = str(
                raw_result.get("url") or ""
            ).strip()

            snippet = str(
                raw_result.get("content") or ""
            ).strip()

            if not title or not url:
                continue

            try:
                normalized_url = self._normalize_url(url)
            except ValueError:
                continue
            # if already exists, skip the url
            if normalized_url in seen_urls:
                continue

            seen_urls.add(normalized_url)

            provider_score = self._safe_score(
                raw_result.get("score")
            )

            normalized_results.append(
                SearchResult(
                    id=generate_search_result_id(),
                    query_id=search_query.id,
                    provider=SourceProvider.TAVILY,
                    canonical_id=f"web:{normalized_url}",
                    title=title,
                    url=url,
                    snippet=snippet or None,
                    provider_score=provider_score,
                )
            )

        return normalized_results

    @staticmethod
    def _normalize_url(
        url: str,
    ) -> str:
        """
        Normalizes URLs for lightweight duplicate detection.

        Security validation is intentionally handled separately by the URL
        guardrail before webpage extraction.
        """

        return url_dedup_key(url)

    @staticmethod
    def _safe_score(
        value: Any,
    ) -> float | None:
        """
        Converts a provider score into a valid 0-to-1 value when possible.
        """

        if value is None:
            return None

        try:
            score = float(value)

        except (TypeError, ValueError):
            return None

        if not 0.0 <= score <= 1.0:
            return None

        return score


    async def close(
        self,
    ) -> None:
        """Closes Tavily's underlying async HTTP client."""

        await self._client.close()


