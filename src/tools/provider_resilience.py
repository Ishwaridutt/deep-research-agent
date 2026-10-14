"""Shared retry and circuit-breaking behavior for external providers."""

import asyncio
from collections.abc import Awaitable, Callable
from time import monotonic
from typing import TypeVar


T = TypeVar("T")


class ProviderOperationError(RuntimeError):
    """Provider failure annotated with whether another provider may be tried."""

    def __init__(self, message: str, *, fallback_allowed: bool = True) -> None:
        super().__init__(message)
        self.fallback_allowed = fallback_allowed


class ProviderCircuitBreaker:
    """Temporarily bypasses a provider after repeated availability failures."""

    def __init__(self, *, failure_threshold: int, recovery_seconds: float) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_seconds = recovery_seconds
        self._failures = 0
        self._open_until = 0.0

    @property
    def is_open(self) -> bool:
        return monotonic() < self._open_until

    def record_success(self) -> None:
        self._failures = 0
        self._open_until = 0.0

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._failure_threshold:
            self._open_until = monotonic() + self._recovery_seconds


async def call_with_retries(
    operation: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    retry_delay_seconds: float,
) -> T:
    """Retry only failures that are eligible for provider failover."""

    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except ProviderOperationError as exc:
            if not exc.fallback_allowed or attempt == max_retries:
                raise
            if retry_delay_seconds:
                await asyncio.sleep(retry_delay_seconds * (2 ** attempt))

    raise RuntimeError("Provider retry loop ended unexpectedly.")


def fallback_allowed_for_status(status_code: int) -> bool:
    """Classify HTTP responses that another provider can reasonably recover from."""

    return status_code in {401, 403, 408, 429} or status_code >= 500


def fallback_allowed_for_exception(exc: Exception) -> bool:
    """Read an HTTP status from SDK exceptions when one is available."""

    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    status_code = status_code or getattr(exc, "status_code", None)
    try:
        if status_code is not None:
            return fallback_allowed_for_status(int(status_code))
    except (TypeError, ValueError):
        pass
    return True
