"""Native Wikipedia search adapter."""

from html import unescape
import re
from typing import Any
from urllib.parse import quote

import httpx

from src.models import SearchQuery, SearchResult, SourceProvider
from src.services import generate_search_result_id
from src.tools.search.base import SearchProvider, SearchProviderError


_HTML_TAG = re.compile(r"<[^>]+>")


class WikipediaSearchProvider(SearchProvider):
    """Searches Wikipedia through the MediaWiki Action API."""

    def __init__(
        self,
        *,
        api_url: str,
        timeout_seconds: int,
        user_agent: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_url = api_url
        self._timeout_seconds = timeout_seconds
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            headers={"User-Agent": user_agent},
        )

    async def search(
        self,
        *,
        search_query: SearchQuery,
        max_results: int,
    ) -> list[SearchResult]:
        try:
            print('Starting Wikipedia search for query: ', search_query.query)
            response = await self._client.get(
                self._api_url,
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": search_query.query,
                    "srnamespace": 0,
                    "srlimit": max_results,
                    "srprop": "snippet|wordcount|timestamp",
                    "format": "json",
                    "formatversion": 2,
                },
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            raw_results = payload.get("query", {}).get("search", [])
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            raise SearchProviderError(
                f"Wikipedia search failed for query '{search_query.id}'."
            ) from exc

        results = [
            result
            for item in raw_results
            if isinstance(item, dict)
            if (result := self._normalize_result(search_query, item))
        ][:max_results]

        print(
            "\nWikipedia search completed:",
            f"query={search_query.id}",
            f"results={len(results)}",
        )
        return results

    @staticmethod
    def _normalize_result(
        search_query: SearchQuery,
        item: dict[str, Any],
    ) -> SearchResult | None:
        title = str(item.get("title") or "").strip()
        page_id = item.get("pageid")
        if not title or not isinstance(page_id, int):
            return None

        article_name = quote(title.replace(" ", "_"), safe="()_-.")
        snippet = unescape(_HTML_TAG.sub("", str(item.get("snippet") or "")))

        return SearchResult(
            id=generate_search_result_id(),
            query_id=search_query.id,
            provider=SourceProvider.WIKIPEDIA,
            canonical_id=f"wikipedia:{page_id}",
            title=title,
            url=f"https://en.wikipedia.org/wiki/{article_name}",
            snippet=" ".join(snippet.split()) or None,
            provider_metadata={
                "page_id": page_id,
                "word_count": item.get("wordcount"),
                "last_edited_at": item.get("timestamp"),
            },
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
