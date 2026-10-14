"""Primary-to-secondary web search failover."""

import asyncio

from src.models import SearchQuery, SearchResult
from src.tools.provider_resilience import (
    ProviderCircuitBreaker,
    call_with_retries,
)
from src.tools.search.base import SearchProvider, SearchProviderError


class FallbackSearchProvider(SearchProvider):
    """Retries the primary search provider before using an optional fallback."""

    def __init__(
        self,
        *,
        primary: SearchProvider,
        fallback: SearchProvider | None,
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

    async def search(
        self,
        *,
        search_query: SearchQuery,
        max_results: int,
    ) -> list[SearchResult]:
        if not self._primary_circuit.is_open:
            try:
                results = await call_with_retries(
                    lambda: self._primary.search(
                        search_query=search_query,
                        max_results=max_results,
                    ),
                    max_retries=self._max_retries,
                    retry_delay_seconds=self._retry_delay_seconds,
                )
            except SearchProviderError as exc:
                if not exc.fallback_allowed:
                    raise
                self._primary_circuit.record_failure()
            else:
                self._primary_circuit.record_success()
                return results

        if self._fallback is None or self._fallback_circuit.is_open:
            raise SearchProviderError("No healthy general search provider is available.")

        # Failover is visible during development without exposing credentials or content.
        print("General search fallback activated:", f"query={search_query.id}")
        try:
            results = await call_with_retries(
                lambda: self._fallback.search(
                    search_query=search_query,
                    max_results=max_results,
                ),
                max_retries=self._max_retries,
                retry_delay_seconds=self._retry_delay_seconds,
            )
        except SearchProviderError as exc:
            if exc.fallback_allowed:
                self._fallback_circuit.record_failure()
            raise

        self._fallback_circuit.record_success()
        return results

    async def close(self) -> None:
        providers = [self._primary]
        if self._fallback is not None:
            providers.append(self._fallback)

        outcomes = await asyncio.gather(
            *(provider.close() for provider in providers),
            return_exceptions=True,
        )
        if any(isinstance(outcome, Exception) for outcome in outcomes):
            raise SearchProviderError("A general search provider failed to close.")
