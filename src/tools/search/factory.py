"""Factory for constructing the configured web search provider."""

from src.config import Settings
from src.tools.search.base import SearchProvider
from src.tools.search.tavily_search import TavilySearchProvider


def create_search_provider(
    settings: Settings,
) -> SearchProvider:
    """
    Creates the configured web search provider.

    Adding another provider later requires a new adapter and one new
    factory branch without changing Research Agent code.
    """

    provider_name = settings.search.search_provider.strip().lower()

    # later we can add more search provider if that is required
    if provider_name == "tavily":
        return TavilySearchProvider(
            api_key=settings.search.tavily_api_key.get_secret_value(),
            timeout_seconds=settings.search.search_timeout_seconds,
        )

    raise ValueError(
        (
            "Unsupported search provider: "
            f"'{settings.search.search_provider}'."
        )
    )


