"""
External tools used by research agents.
Provider-independent web search tools.
"""

from src.tools.search.base import (
    SearchProvider,
    SearchProviderError,
)
from src.tools.search.factory import (
    create_search_provider,
)
from src.tools.search.arxiv_search import ArxivSearchProvider
from src.tools.search.wikipedia_search import WikipediaSearchProvider
from src.tools.search.brave_search import BraveSearchProvider
from src.tools.search.fallback_search import FallbackSearchProvider
from src.tools.search.parallel_search import (
    ParallelSearchProvider,
    deduplicate_search_results,
)

__all__ = [
    "SearchProvider",
    "SearchProviderError",
    "ArxivSearchProvider",
    "BraveSearchProvider",
    "FallbackSearchProvider",
    "ParallelSearchProvider",
    "WikipediaSearchProvider",
    "create_search_provider",
    "deduplicate_search_results",
]
