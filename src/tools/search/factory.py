"""Factory for constructing the configured web search provider."""

from src.config import Settings
from src.tools.search.base import SearchProvider
from src.tools.search.arxiv_search import ArxivSearchProvider
from src.tools.search.brave_search import BraveSearchProvider
from src.tools.search.fallback_search import FallbackSearchProvider
from src.tools.search.parallel_search import ParallelSearchProvider
from src.tools.search.tavily_search import TavilySearchProvider
from src.tools.search.wikipedia_search import WikipediaSearchProvider


def create_search_provider(
    settings: Settings,
) -> SearchProvider:
    """
    Creates the configured parallel public search provider.
    """

    search = settings.search
    provider_name = search.search_provider.strip().lower()

    if provider_name == "tavily":
        primary_provider = TavilySearchProvider(
            api_key=search.tavily_api_key.get_secret_value(),
            timeout_seconds=search.search_timeout_seconds,
        )
    else:
        raise ValueError(
            f"Unsupported search provider: '{search.search_provider}'."
        )

    brave_key = (
        search.brave_api_key.get_secret_value().strip()
        if search.brave_api_key
        else ""
    )
    fallback_provider = (
        BraveSearchProvider(
            api_key=brave_key,
            api_url=search.brave_api_url,
            api_version=search.brave_api_version,
            timeout_seconds=search.search_timeout_seconds,
        )
        if brave_key
        else None
    )

    # The graph sees one general-search branch; retry, circuit state, and Brave
    # activation remain internal to that branch.
    general_provider = FallbackSearchProvider(
        primary=primary_provider,
        fallback=fallback_provider,
        max_retries=search.provider_max_retries,
        retry_delay_seconds=search.provider_retry_delay_seconds,
        circuit_failure_threshold=search.provider_circuit_failure_threshold,
        circuit_recovery_seconds=search.provider_circuit_recovery_seconds,
    )
    branches: list[tuple[str, SearchProvider, int]] = [
        ("web", general_provider, search.web_results_per_query),
    ]

    # Disabled native providers are omitted entirely, so they create no clients
    # and consume no search or shutdown work.
    if search.arxiv_enabled:
        branches.append((
            "arxiv",
            ArxivSearchProvider(
                api_url=search.arxiv_api_url,
                timeout_seconds=search.arxiv_timeout_seconds,
                user_agent=search.public_api_user_agent,
            ),
            search.arxiv_results_per_query,
        ))
    if search.wikipedia_enabled:
        branches.append((
            "wikipedia",
            WikipediaSearchProvider(
                api_url=search.wikipedia_api_url,
                timeout_seconds=search.wikipedia_timeout_seconds,
                user_agent=search.public_api_user_agent,
            ),
            search.wikipedia_results_per_query,
        ))

    return ParallelSearchProvider(
        branches=branches,
        max_concurrency=search.search_max_concurrency,
    )


