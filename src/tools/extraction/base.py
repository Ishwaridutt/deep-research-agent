"""Provider-independent webpage extraction contracts."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class ExtractedWebPage:
    """Normalized content returned by a webpage extraction provider."""

    url: str
    content: str


class WebExtractionError(RuntimeError):
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

        