"""Client for TypeSafe's Jev decision model (POST /v1/systemone)."""

import asyncio
import logging
from functools import partial
from time import monotonic

import httpx

from src.tools.provider_resilience import ProviderOperationError, call_with_retries


logger = logging.getLogger(__name__)

# Each request asks one question; its answer comes back under this name.
_QUESTION_NAME = "answer"


class JevServiceError(RuntimeError):
    """Jev could not answer: network failure, HTTP error, or an unexpected response."""


class JevService:
    """
    Asks Jev typed questions about a state and returns calibrated probabilities.

    One shared HTTP client and semaphore bound concurrent requests for the whole
    process. Timeouts, 408, 429, and 5xx responses are retried with backoff; any
    remaining failure raises JevServiceError so callers can fall back to an LLM.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        api_url: str,
        timeout_seconds: float,
        max_concurrency: int,
        max_retries: int,
        retry_delay_seconds: float,
        private_data_allowed: bool = False,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._api_url = api_url
        self._max_retries = max_retries
        self._retry_delay_seconds = retry_delay_seconds
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._client = client or httpx.AsyncClient(timeout=timeout_seconds)
        # True only when LLM_PRIVATE_DATA_PROVIDERS lists "typesafe".
        self.private_data_allowed = private_data_allowed

    async def noul(self, *, state: object, instructions: str, criteria: dict[str, str] | None = None) -> float:
        """Return the probability (0 to 1) that the yes/no question is true for the state."""
        question = {"type": "noul", "instructions": instructions}
        if criteria:
            question["criteria"] = criteria
        answer = await self._ask(state=state, question=question)
        try:
            return min(1.0, max(0.0, float(answer["noul"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise JevServiceError("Jev returned an invalid noul answer.") from exc

    async def choice(
        self,
        *,
        state: object,
        instructions: str,
        criteria: dict[str, str],
    ) -> tuple[str, float, dict[str, float]]:
        """Return (top choice, confidence, probability of every choice)."""
        answer = await self._ask(
            state=state,
            question={"type": "choice", "instructions": instructions, "criteria": criteria},
        )
        try:
            choice = str(answer["choice"])
            confidence = float(answer["confidence"])
            probabilities = {str(name): float(value) for name, value in answer["probabilities"].items()}
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise JevServiceError("Jev returned an invalid choice answer.") from exc
        if choice not in criteria:
            raise JevServiceError(f"Jev returned an unknown choice '{choice}'.")
        return choice, confidence, probabilities

    async def close(self) -> None:
        await self._client.aclose()

    async def _ask(self, *, state: object, question: dict) -> dict:
        body = {"model": self._model, "state": state, "questions": {_QUESTION_NAME: question}}
        started = monotonic()
        try:
            async with self._semaphore:
                payload = await call_with_retries(
                    partial(self._post, body),
                    max_retries=self._max_retries,
                    retry_delay_seconds=self._retry_delay_seconds,
                )
        except ProviderOperationError as exc:
            logger.warning(
                "Jev request failed.",
                extra={"event": "jev_request_failed", "node": "jev_service", "data": {"error": str(exc)}},
            )
            raise JevServiceError(str(exc)) from exc

        try:
            answer = payload["answers"][_QUESTION_NAME]
        except (KeyError, TypeError) as exc:
            raise JevServiceError("Jev response has no answer for the question.") from exc
        usage = payload.get("usage") or {}
        logger.info(
            "Jev request completed.",
            extra={
                "event": "jev_request_completed",
                "node": "jev_service",
                "data": {
                    "model": payload.get("model", self._model),
                    "question_type": question["type"],
                    "latency_ms": round((monotonic() - started) * 1000),
                    "input_tokens": usage.get("input_tokens"),
                },
            },
        )
        return answer

    async def _post(self, body: dict) -> dict:
        """Send one request; retryable failures are marked for call_with_retries."""
        try:
            response = await self._client.post(
                self._api_url,
                json=body,
                headers={"Authorization": f"Bearer {self._api_key}"},
            )
        except httpx.HTTPError as exc:
            raise ProviderOperationError(f"Jev request failed: {type(exc).__name__}.") from exc

        if response.status_code != 200:
            status = response.status_code
            raise ProviderOperationError(
                f"Jev returned HTTP {status}.",
                fallback_allowed=status in {408, 429} or status >= 500,
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ProviderOperationError("Jev returned invalid JSON.", fallback_allowed=False) from exc
