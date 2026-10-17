"""Brave implementation of the SearchProvider contract."""

from typing import Any

import httpx

from src.guardrails import url_dedup_key
from src.models import SearchQuery, SearchResult, SourceProvider
from src.services import generate_search_result_id
from src.tools.provider_resilience import fallback_allowed_for_status
from src.tools.search.base import SearchProvider, SearchProviderError


class BraveSearchProvider(SearchProvider):
    """Searches Brave's web index and normalizes its ranked results."""

    def __init__(
        self,
        *,
        api_key: str,
        api_url: str,
        api_version: str,
        timeout_seconds: int,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_url = api_url
        self._api_version = api_version
        self._timeout_seconds = timeout_seconds
        self._owns_client = client is None
        self._headers = {
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        }
        self._client = client or httpx.AsyncClient()

    async def search(
        self,
        *,
        search_query: SearchQuery,
        max_results: int,
    ) -> list[SearchResult]:
        headers = dict(self._headers)
        if self._api_version:
            headers["Api-Version"] = self._api_version
        try:
            print('Starting Brave search for query: ', search_query.query)
            response = await self._client.get(
                self._api_url,
                params={
                    "q": search_query.query,
                    "count": min(max_results, 20),
                    "safesearch": "moderate",
                },
                headers=headers,
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise SearchProviderError(
                f"Brave search failed for query '{search_query.id}'.",
                fallback_allowed=fallback_allowed_for_status(exc.response.status_code),
            ) from exc
        except (httpx.RequestError, ValueError) as exc:
            raise SearchProviderError(
                f"Brave search failed for query '{search_query.id}'."
            ) from exc

        if not isinstance(payload, dict):
            raise SearchProviderError(
                f"Brave returned invalid data for query '{search_query.id}'."
            )
        web = payload.get("web") or {}
        if not isinstance(web, dict):
            raise SearchProviderError(
                f"Brave returned invalid results for query '{search_query.id}'."
            )
        raw_results = web.get("results", [])
        if not isinstance(raw_results, list):
            raise SearchProviderError(
                f"Brave returned invalid results for query '{search_query.id}'."
            )
        return self._normalize_results(
            search_query=search_query,
            raw_results=raw_results,
            max_results=max_results,
        )

    @staticmethod
    def _normalize_results(
        *,
        search_query: SearchQuery,
        raw_results: list[dict[str, Any]],
        max_results: int,
    ) -> list[SearchResult]:
        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            if not title or not url:
                continue

            try:
                normalized_url = url_dedup_key(url)
            except ValueError:
                continue
            if normalized_url in seen_urls:
                continue

            seen_urls.add(normalized_url)
            results.append(SearchResult(
                id=generate_search_result_id(),
                query_id=search_query.id,
                provider=SourceProvider.BRAVE,
                canonical_id=f"web:{normalized_url}",
                title=title,
                url=url,
                snippet=str(item.get("description") or "").strip() or None,
                provider_metadata={
                    "age": item.get("age"),
                    "language": item.get("language"),
                },
            ))
            if len(results) >= max_results:
                break
        return results

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
