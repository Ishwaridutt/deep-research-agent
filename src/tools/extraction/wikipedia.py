"""Wikipedia article content extractor."""

import httpx

from src.models import SearchResult, SourceProvider
from src.tools.extraction.base import ExtractedSourceContent, SourceExtractionError


class WikipediaContentExtractor:
    """Extracts canonical plain-text articles through MediaWiki."""

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

    async def extract(
        self,
        *,
        search_result: SearchResult,
    ) -> ExtractedSourceContent:
        if search_result.provider != SourceProvider.WIKIPEDIA:
            raise SourceExtractionError("Expected a Wikipedia search result.")

        page_id = search_result.provider_metadata.get("page_id")
        if not isinstance(page_id, int):
            raise SourceExtractionError("Wikipedia result has no page ID.")

        try:
            response = await self._client.get(
                self._api_url,
                params={
                    "action": "query",
                    "pageids": page_id,
                    "redirects": 1,
                    "prop": "extracts|info|revisions",
                    "explaintext": 1,
                    "exsectionformat": "plain",
                    "inprop": "url",
                    "rvprop": "ids|timestamp",
                    "format": "json",
                    "formatversion": 2,
                },
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            pages = payload.get("query", {}).get("pages", [])
            page = pages[0] if pages else {}
        except (httpx.HTTPError, ValueError, AttributeError) as exc:
            raise SourceExtractionError("Wikipedia article extraction failed.") from exc

        content = str(page.get("extract") or "").strip()
        if page.get("missing") or not content:
            raise SourceExtractionError("Wikipedia article contains no usable text.")

        revisions = page.get("revisions") or []
        revision = revisions[0] if revisions else {}
        metadata = {
            **search_result.provider_metadata,
            "page_id": page.get("pageid", page_id),
            "revision_id": revision.get("revid"),
            "revision_timestamp": revision.get("timestamp"),
            "redirects": payload.get("query", {}).get("redirects", []),
        }
        url = str(page.get("canonicalurl") or search_result.url)

        print(
            "Wikipedia article loaded:",
            f"page_id={metadata['page_id']}",
            f"characters={len(content)}",
        )
        return ExtractedSourceContent(
            url=url,
            content=content,
            content_format="text",
            provider_metadata=metadata,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
