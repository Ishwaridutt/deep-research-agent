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

__all__ = [
    "SearchProvider",
    "SearchProviderError",
    "create_search_provider",
]
