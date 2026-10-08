"""Factory for constructing the configured webpage extractor."""

from src.config import Settings
from src.tools.extraction.base import (
    WebExtractor,
)
from src.tools.extraction.tavily import (
    TavilyWebExtractor,
)


def create_web_extractor(
    settings: Settings,
) -> WebExtractor:
    """Creates the configured webpage extraction provider."""

    provider_name = settings.search.web_extractor_provider.strip().lower()

    if provider_name == "tavily":
        return TavilyWebExtractor(
            api_key=settings.search.tavily_api_key.get_secret_value(),
            timeout_seconds=settings.search.extraction_timeout_seconds,
            extract_depth=settings.search.tavily_extract_depth,
            output_format=settings.search.tavily_extract_format,
        )

    raise ValueError(
        (
            "Unsupported webpage extractor: "
            f"'{settings.search.web_extractor_provider}'."
        )
    )

