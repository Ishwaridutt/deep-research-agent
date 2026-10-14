"""Native arXiv search adapter."""

import re
from xml.etree import ElementTree

import httpx

from src.models import SearchQuery, SearchResult, SourceProvider
from src.services import generate_search_result_id
from src.tools.search.base import SearchProvider, SearchProviderError


_ATOM = "http://www.w3.org/2005/Atom"
_ARXIV = "http://arxiv.org/schemas/atom"
_NAMESPACES = {"atom": _ATOM, "arxiv": _ARXIV}
_VERSION_PATTERN = re.compile(r"^(?P<id>.+?)(?P<version>v\d+)?$")


class ArxivSearchProvider(SearchProvider):
    """Searches arXiv's Atom API and normalizes paper metadata."""

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
            response = await self._client.get(
                self._api_url,
                params={
                    "search_query": f"all:{search_query.query}",
                    "start": 0,
                    "max_results": max_results,
                    "sortBy": "relevance",
                    "sortOrder": "descending",
                },
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
        except (httpx.HTTPError, ElementTree.ParseError) as exc:
            raise SearchProviderError(
                f"arXiv search failed for query '{search_query.id}'."
            ) from exc

        results = [
            result
            for entry in root.findall("atom:entry", _NAMESPACES)
            if (result := self._normalize_entry(search_query, entry))
        ][:max_results]

        print(
            "arXiv search completed:",
            f"query={search_query.id}",
            f"results={len(results)}",
        )
        return results

    @staticmethod
    def _normalize_entry(
        search_query: SearchQuery,
        entry: ElementTree.Element,
    ) -> SearchResult | None:
        entry_id = _text(entry, "atom:id")
        title = _text(entry, "atom:title")
        summary = _text(entry, "atom:summary")
        if not entry_id or not title:
            return None

        if "/abs/" not in entry_id:
            return None

        paper_id = entry_id.rsplit("/abs/", 1)[-1].strip()
        match = _VERSION_PATTERN.fullmatch(paper_id)
        if not match:
            return None

        base_id = match.group("id")
        version = match.group("version")
        canonical_id = f"arxiv:{base_id}"
        if version:
            canonical_id = f"{canonical_id}:{version}"

        links = entry.findall("atom:link", _NAMESPACES)
        pdf_url = next(
            (
                link.get("href")
                for link in links
                if link.get("type") == "application/pdf"
            ),
            f"https://arxiv.org/pdf/{paper_id}",
        )
        authors = [
            name
            for author in entry.findall("atom:author", _NAMESPACES)
            if (name := _text(author, "atom:name"))
        ]
        categories = [
            term
            for category in entry.findall("atom:category", _NAMESPACES)
            if (term := category.get("term"))
        ]

        metadata = {
            "arxiv_id": base_id,
            "version": version,
            "authors": authors,
            "abstract": summary,
            "published_at": _text(entry, "atom:published"),
            "updated_at": _text(entry, "atom:updated"),
            "categories": categories,
            "primary_category": _attribute(
                entry,
                "arxiv:primary_category",
                "term",
            ),
            "doi": _text(entry, "arxiv:doi"),
            "journal_reference": _text(entry, "arxiv:journal_ref"),
            "pdf_url": pdf_url,
        }

        return SearchResult(
            id=generate_search_result_id(),
            query_id=search_query.id,
            provider=SourceProvider.ARXIV,
            canonical_id=canonical_id,
            title=title,
            url=f"https://arxiv.org/abs/{paper_id}",
            snippet=summary or None,
            provider_metadata=metadata,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _text(
    element: ElementTree.Element,
    path: str,
) -> str:
    child = element.find(path, _NAMESPACES)
    return " ".join((child.text or "").split()) if child is not None else ""


def _attribute(
    element: ElementTree.Element,
    path: str,
    name: str,
) -> str | None:
    child = element.find(path, _NAMESPACES)
    return child.get(name) if child is not None else None
