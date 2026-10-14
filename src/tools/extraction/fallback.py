"""Primary-to-secondary webpage extraction failover."""

import asyncio

from src.tools.extraction.base import (
    ExtractedWebPage,
    WebExtractionError,
    WebExtractor,
)
from src.tools.provider_resilience import (
    ProviderCircuitBreaker,
    call_with_retries,
)


class FallbackWebExtractor(WebExtractor):
    """Retries the primary extractor before using an optional fallback."""

    def __init__(
        self,
        *,
        primary: WebExtractor,
        fallback: WebExtractor | None,
        max_retries: int,
        retry_delay_seconds: float,
        circuit_failure_threshold: int,
        circuit_recovery_seconds: float,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._max_retries = max_retries
        self._retry_delay_seconds = retry_delay_seconds
        self._primary_circuit = ProviderCircuitBreaker(
            failure_threshold=circuit_failure_threshold,
            recovery_seconds=circuit_recovery_seconds,
        )
        self._fallback_circuit = ProviderCircuitBreaker(
            failure_threshold=circuit_failure_threshold,
            recovery_seconds=circuit_recovery_seconds,
        )

    async def extract(self, *, url: str) -> ExtractedWebPage:
        if not self._primary_circuit.is_open:
            try:
                page = await call_with_retries(
                    lambda: self._primary.extract(url=url),
                    max_retries=self._max_retries,
                    retry_delay_seconds=self._retry_delay_seconds,
                )
            except WebExtractionError as exc:
                if not exc.fallback_allowed:
                    raise
                self._primary_circuit.record_failure()
            else:
                self._primary_circuit.record_success()
                return page

        if self._fallback is None or self._fallback_circuit.is_open:
            raise WebExtractionError("No healthy webpage extractor is available.")

        print("Web extraction fallback activated:", f"url={url}")
        try:
            page = await call_with_retries(
                lambda: self._fallback.extract(url=url),
                max_retries=self._max_retries,
                retry_delay_seconds=self._retry_delay_seconds,
            )
        except WebExtractionError as exc:
            if exc.fallback_allowed:
                self._fallback_circuit.record_failure()
            raise

        self._fallback_circuit.record_success()
        return page

    async def close(self) -> None:
        extractors = [self._primary]
        if self._fallback is not None:
            extractors.append(self._fallback)

        outcomes = await asyncio.gather(
            *(extractor.close() for extractor in extractors),
            return_exceptions=True,
        )
        if any(isinstance(outcome, Exception) for outcome in outcomes):
            raise WebExtractionError("A webpage extractor failed to close.")
