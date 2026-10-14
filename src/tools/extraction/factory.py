"""Factories for configured public-source extractors."""

from src.config import Settings
from src.documents.document_parser import DocumentParser
from src.tools.extraction.arxiv import ArxivContentExtractor
from src.tools.extraction.base import (
    SourceContentExtractor,
    WebExtractor,
)
from src.tools.extraction.fallback import FallbackWebExtractor
from src.tools.extraction.firecrawl import FirecrawlWebExtractor
from src.tools.extraction.tavily import (
    TavilyWebExtractor,
)
from src.tools.extraction.source_extractor import ProviderAwareSourceExtractor
from src.tools.extraction.wikipedia import WikipediaContentExtractor


def create_web_extractor(
    settings: Settings,
) -> WebExtractor:
    """Creates the configured webpage extraction provider."""

    provider_name = settings.search.web_extractor_provider.strip().lower()

    if provider_name == "tavily":
        primary_extractor = TavilyWebExtractor(
            api_key=settings.search.tavily_api_key.get_secret_value(),
            timeout_seconds=settings.search.extraction_timeout_seconds,
            extract_depth=settings.search.tavily_extract_depth,
            output_format=settings.search.tavily_extract_format,
        )

    else:
        raise ValueError(
            "Unsupported webpage extractor: "
            f"'{settings.search.web_extractor_provider}'."
        )

    search = settings.search
    firecrawl_key = (
        search.firecrawl_api_key.get_secret_value().strip()
        if search.firecrawl_api_key
        else ""
    )
    fallback_extractor = (
        FirecrawlWebExtractor(
            api_key=firecrawl_key,
            api_url=search.firecrawl_api_url,
            timeout_seconds=search.extraction_timeout_seconds,
        )
        if firecrawl_key
        else None
    )

    # The source processor keeps one extractor contract while this wrapper owns
    # retries, circuit state, and optional Firecrawl activation.
    return FallbackWebExtractor(
        primary=primary_extractor,
        fallback=fallback_extractor,
        max_retries=search.provider_max_retries,
        retry_delay_seconds=search.provider_retry_delay_seconds,
        circuit_failure_threshold=search.provider_circuit_failure_threshold,
        circuit_recovery_seconds=search.provider_circuit_recovery_seconds,
    )


def create_source_content_extractor(settings: Settings) -> SourceContentExtractor:
    """Create one router over general-web and native content extractors."""

    search = settings.search
    arxiv = (
        ArxivContentExtractor(
            parser=DocumentParser(),
            timeout_seconds=search.arxiv_timeout_seconds,
            max_pdf_bytes=settings.research_limits.max_source_download_bytes,
            max_pdf_pages=search.arxiv_max_pdf_pages,
            full_text_enabled=search.arxiv_full_text_enabled,
            user_agent=search.public_api_user_agent,
        )
        if search.arxiv_enabled
        else None
    )
    wikipedia = (
        WikipediaContentExtractor(
            api_url=search.wikipedia_api_url,
            timeout_seconds=search.wikipedia_timeout_seconds,
            user_agent=search.public_api_user_agent,
        )
        if search.wikipedia_enabled
        else None
    )
    return ProviderAwareSourceExtractor(
        web=create_web_extractor(settings),
        arxiv=arxiv,
        wikipedia=wikipedia,
    )

