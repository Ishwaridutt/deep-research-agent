"""Provider-independent source extraction contracts."""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from src.models import SearchResult
from src.tools.provider_resilience import ProviderOperationError


@dataclass(slots=True)
class ExtractedWebPage:
    """Normalized content returned by a webpage extraction provider."""

    url: str
    content: str
    content_format: Literal["text", "markdown", "html"] = "text"
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExtractedSourceContent:
    """Normalized content returned by a provider-aware extractor."""

    url: str
    content: str
    content_format: Literal["text", "markdown", "html"] = "text"
    provider_metadata: dict[str, Any] = field(default_factory=dict)


class SourceExtractionError(ProviderOperationError):
    """Raised when a public source cannot provide usable content."""


class WebExtractionError(ProviderOperationError):
    """Raised when webpage content cannot be extracted."""


class WebExtractor(Protocol):
    """Contract implemented by webpage extraction providers."""

    async def extract(
        self,
        *,
        url: str,
    ) -> ExtractedWebPage:
        """Extracts normalized content from one webpage."""

        ...

    async def close(
        self,
    ) -> None:
        """Releases resources owned by the extractor."""

        ...


class SourceContentExtractor(Protocol):
    """Extracts full content for one normalized public search result."""

    async def extract(
        self,
        *,
        search_result: SearchResult,
    ) -> ExtractedSourceContent:
        ...

    async def close(self) -> None:
        ...
