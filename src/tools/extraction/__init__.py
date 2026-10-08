"""Provider-independent webpage extraction tools."""

from src.tools.extraction.base import (
    ExtractedWebPage,
    WebExtractionError,
    WebExtractor,
)
from src.tools.extraction.factory import (
    create_web_extractor,
)

__all__ = [
    "ExtractedWebPage",
    "WebExtractionError",
    "WebExtractor",
    "create_web_extractor",
]

