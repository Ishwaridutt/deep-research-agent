"""Provider-independent web search contracts."""

from typing import Protocol
from src.models import SearchQuery, SearchResult

class SearchProviderError(RuntimeError):
    """Raised when a web search provider cannot complete a request."""


class SearchProvider(Protocol):
    """
    Contract implemented by web search providers.

    Research Agent code depends on this interface rather than directly
    depending on Tavily or another external search service.
    """

    async def search(
        self,
        *,
        search_query: SearchQuery,
        max_results: int,
    ) -> list[SearchResult]:
        """
        Searches the web and returns normalized search results.
        """

        ...


    async def close(
    self,
    ) -> None:
        """Releases resources owned by the search provider."""

        ...


