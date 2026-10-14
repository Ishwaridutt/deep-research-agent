"""Routes normalized search results to their content extractor."""

import asyncio

from src.models import SearchResult, SourceProvider
from src.tools.extraction.base import (
    ExtractedSourceContent,
    SourceContentExtractor,
    SourceExtractionError,
    WebExtractionError,
    WebExtractor,
)


class ProviderAwareSourceExtractor:
    """Uses native extractors when available and web extraction otherwise."""

    def __init__(
        self,
        *,
        web: WebExtractor,
        arxiv: SourceContentExtractor | None,
        wikipedia: SourceContentExtractor | None,
    ) -> None:
        self._web = web
        self._native = {
            provider: extractor
            for provider, extractor in (
                (SourceProvider.ARXIV, arxiv),
                (SourceProvider.WIKIPEDIA, wikipedia),
            )
            if extractor is not None
        }

    async def extract(
        self,
        *,
        search_result: SearchResult,
    ) -> ExtractedSourceContent:
        native_extractor = self._native.get(search_result.provider)
        if native_extractor is not None:
            return await native_extractor.extract(search_result=search_result)

        if search_result.provider not in {SourceProvider.TAVILY, SourceProvider.BRAVE}:
            raise SourceExtractionError(
                f"No content extractor is configured for '{search_result.provider.value}'.",
                fallback_allowed=False,
            )

        # General search providers discover URLs; Tavily/Firecrawl fetch their content.
        try:
            page = await self._web.extract(url=search_result.url)
        except WebExtractionError as exc:
            raise SourceExtractionError(
                str(exc),
                fallback_allowed=exc.fallback_allowed,
            ) from exc

        return ExtractedSourceContent(
            url=page.url,
            content=page.content,
            content_format=page.content_format,
            provider_metadata=page.provider_metadata,
        )

    async def close(self) -> None:
        # Close every owned adapter even if another adapter's cleanup fails.
        extractors = [self._web, *self._native.values()]
        outcomes = await asyncio.gather(
            *(extractor.close() for extractor in extractors),
            return_exceptions=True,
        )
        if any(isinstance(outcome, Exception) for outcome in outcomes):
            raise SourceExtractionError("A source content extractor failed to close.")
