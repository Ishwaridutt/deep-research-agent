"""Shared service for invoking provider-independent LLMs."""

import asyncio
import json
import logging

from dataclasses import dataclass
from typing import Literal, Sequence, TypeVar

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ValidationError

from src.llm.roles import LLMRole


logger = logging.getLogger(__name__)

StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)
StructuredOutputMethod = Literal["json_schema", "function_calling"]


@dataclass(slots=True)
class LLMCallMetadata:
    """Metadata attached to an LLM call for tracing and observability."""

    job_id: str
    research_session_id: str
    node: str
    iteration: int | None = None
    contains_private_data: bool = False

    def as_dict(self) -> dict[str, str | int]:
        """Convert metadata into a LangChain-compatible dictionary."""

        metadata: dict[str, str | int] = {
            "job_id": self.job_id,
            "research_session_id": self.research_session_id,
            "node": self.node,
        }

        if self.iteration is not None:
            metadata["iteration"] = self.iteration

        metadata["contains_private_data"] = self.contains_private_data

        return metadata


class LLMServiceError(RuntimeError):
    """Raised when an LLM invocation fails."""

    def __init__(self, *, role: LLMRole, message: str) -> None:
        self.role = role
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class LLMProviderModel:
    """One provider/model entry in a role's ordered fallback chain."""

    provider: str
    model: BaseChatModel


@dataclass(slots=True)
class _LLMInvocationFailure(Exception):
    reason: str
    fallback_allowed: bool
    provider_unavailable: bool = False


def _provider_error_body(exc: Exception) -> dict:
    """Return structured provider error metadata without generated content."""

    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return {}

    provider_error = body.get("error", body)
    return provider_error if isinstance(provider_error, dict) else {}


def _model_name(model: BaseChatModel) -> str | None:
    """Configured model name for logs (not a secret), so a misconfigured value is visible."""
    name = getattr(model, "model_name", None) or getattr(model, "model", None)
    return str(name) if name else None


def _is_retryable_output_error(exc: Exception) -> bool:
    """Retry malformed structured output, not provider or quota failures."""

    if isinstance(exc, (OutputParserException, ValidationError, json.JSONDecodeError)):
        return True

    if _provider_error_body(exc).get("code") in {
        "json_validate_failed",
        "tool_use_failed",
    }:
        return True

    message = str(exc).lower()

    non_retryable = (
        "rate limit",
        "rate_limit",
        "quota exceeded",
        "resource_exhausted",
        "tpm limit",
        "request too large",
        "context length",
        "authentication",
        "invalid api key",
    )

    if any(reason in message for reason in non_retryable):
        return False

    retryable = (
        "failed to parse tool call arguments as json",
        "failed to call a function",
        "generated json does not match the expected schema",
        "invalid json",
        "json decode",
        "could not parse",
        "output parsing",
        "tool_use_failed",
        "unexpected structured output type",
    )

    return any(reason in message for reason in retryable)


def _provider_fallback_allowed(exc: Exception) -> bool:
    """Fail over availability errors while keeping invalid requests terminal."""

    if _is_retryable_output_error(exc):
        return True

    # A model name one provider does not serve is a configuration problem for
    # that provider only; the next provider in the chain may still work.
    if _provider_error_body(exc).get("code") == "model_not_found":
        return True

    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None) or getattr(exc, "status_code", None)
    try:
        if status is not None:
            status = int(status)
            return status in {401, 403, 404, 408, 409, 429} or status >= 500
    except (TypeError, ValueError):
        pass

    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    message = str(exc).casefold()
    terminal_reasons = (
        "context length",
        "request too large",
        "invalid prompt",
        "invalid_request",
        "content policy",
    )
    if any(reason in message for reason in terminal_reasons):
        return False

    availability_reasons = (
        "unavailable",
        "timeout",
        "timed out",
        "connection",
        "rate limit",
        "rate_limit",
        "quota",
        "resource_exhausted",
        "authentication",
        "api key",
        "server error",
    )
    return any(reason in message for reason in availability_reasons)


def _safe_provider_error_details(exc: Exception) -> dict[str, str | int]:
    """Expose provider diagnostics without logging prompts or generated output."""

    details: dict[str, str | int] = {
        "provider_exception_type": type(exc).__name__,
    }
    status = getattr(exc, "status_code", None)
    try:
        if status is not None:
            details["provider_status_code"] = int(status)
    except (TypeError, ValueError):
        pass

    provider_error = _provider_error_body(exc)
    code = provider_error.get("code")
    category = provider_error.get("type")
    if isinstance(code, (str, int)):
        details["provider_error_code"] = code
    if isinstance(category, str):
        details["provider_error_category"] = category

    return details


class LLMService:
    """Provider-independent interface used by all LLM-backed nodes."""

    def __init__(
        self,
        models: dict[LLMRole, BaseChatModel | list[LLMProviderModel]],
        *,
        structured_output_method: StructuredOutputMethod = "json_schema",
        structured_output_attempts: int = 2,
        private_data_providers: set[str] | None = None,
    ) -> None:
        self._models = {
            role: (
                configured
                if isinstance(configured, list)
                else [LLMProviderModel(provider="configured", model=configured)]
            )
            for role, configured in models.items()
        }
        self._structured_output_method = structured_output_method
        self._structured_output_attempts = max(1, structured_output_attempts)
        self._private_data_providers = private_data_providers
        # self._job_provider: dict[str, str] = {}
        # one provider per job and role
        self._job_provider: dict[str, dict[LLMRole, str]] = {}
        self._job_failed_providers: dict[str, set[str]] = {}

    def _get_models(
        self,
        role: LLMRole,
        metadata: LLMCallMetadata,
    ) -> list[LLMProviderModel]:
        """Return healthy, policy-eligible models in job-preferred order."""

        models = self._models.get(role)

        if not models:
            raise LLMServiceError(
                role=role,
                message=f"No LLM model configured for role '{role.value}'.",
            )

        policy_eligible = models
        if metadata.contains_private_data and self._private_data_providers is not None:
            policy_eligible = [
                model for model in policy_eligible
                if model.provider in self._private_data_providers
            ]
            if not policy_eligible:
                raise LLMServiceError(
                    role=role,
                    message=(
                        "No configured LLM provider is permitted to process "
                        f"private data for role '{role.value}'."
                    ),
                )

        failed = self._job_failed_providers.get(metadata.job_id, set())
        eligible = [
            model for model in policy_eligible if model.provider not in failed
        ]
        # provider lookup
        # preferred = self._job_provider.get(metadata.job_id)
        preferred = self._job_provider.get(metadata.job_id, {}).get(role)
        if preferred:
            eligible.sort(key=lambda item: item.provider != preferred)

        if not eligible:
            raise LLMServiceError(
                role=role,
                message=(
                    "All configured LLM providers are currently unavailable "
                    f"for role '{role.value}'."
                ),
            )
        return eligible

    def release_job(self, job_id: str) -> None:
        """Release provider health and affinity retained for a completed job."""

        self._job_provider.pop(job_id, None)
        self._job_failed_providers.pop(job_id, None)

    async def invoke_structured(
        self,
        *,
        role: LLMRole,
        messages: Sequence[BaseMessage],
        output_schema: type[StructuredOutput],
        metadata: LLMCallMetadata,
    ) -> StructuredOutput:
        """Invoke an LLM with schema validation and bounded parsing retries."""

        models = self._get_models(role, metadata)

        last_failure: _LLMInvocationFailure | None = None
        unavailable_providers: set[str] = set()
        for index, provider_model in enumerate(models):
            try:
                result = await self._invoke_provider(
                    role=role,
                    provider_model=provider_model,
                    messages=messages,
                    output_schema=output_schema,
                    metadata=metadata,
                )
            except _LLMInvocationFailure as failure:
                last_failure = failure
                if not failure.fallback_allowed:
                    raise LLMServiceError(
                        role=role,
                        message=(
                            f"LLM invocation failed for role '{role.value}': "
                            f"{failure.reason}."
                        ),
                    ) from None

                if failure.provider_unavailable:
                    unavailable_providers.add(provider_model.provider)
                if index + 1 < len(models):
                    logger.warning(
                        "LLM provider failed; trying the next eligible provider.",
                        extra={
                            "event": "llm_provider_fallback",
                            **metadata.as_dict(),
                            "role": role.value,
                            "provider": provider_model.provider,
                        },
                    )
                    print(
                        "LLM fallback activated:",
                        f"role={role.value}",
                        f"failed_provider={provider_model.provider}",
                    )
                continue

            # self._job_provider[metadata.job_id] = provider_model.provider
            self._job_provider.setdefault(metadata.job_id, {})[role] = (provider_model.provider)
            if unavailable_providers:
                # Persist availability failures only after another provider succeeds.
                # If every provider fails, the next role may retry them from scratch.
                self._job_failed_providers.setdefault(metadata.job_id, set()).update(
                    unavailable_providers
                )
            logger.info(
                "LLM provider request completed.",
                extra={
                    "event": "llm_provider_succeeded",
                    **metadata.as_dict(),
                    "role": role.value,
                    "provider": provider_model.provider,
                },
            )
            print(
                "LLM provider selected:",
                f"role={role.value}",
                f"provider={provider_model.provider}",
            )
            return result

        raise LLMServiceError(
            role=role,
            message=(
                f"All eligible LLM providers failed for role '{role.value}': "
                f"{last_failure.reason if last_failure else 'provider request failed'}."
            ),
        )

    async def _invoke_provider(
        self,
        *,
        role: LLMRole,
        provider_model: LLMProviderModel,
        messages: Sequence[BaseMessage],
        output_schema: type[StructuredOutput],
        metadata: LLMCallMetadata,
    ) -> StructuredOutput:
        """Invoke one provider and classify whether the next provider may run."""

        try:
            options = {}
            if (
                provider_model.provider == "groq"
                and role in {
                    LLMRole.EVIDENCE_EXTRACTOR,
                    LLMRole.CITATION_CHECKER,
                }
                and self._structured_output_method == "json_schema"
            ):
                # GPT-OSS supports constrained decoding, avoiding best-effort
                # JSON that can fail only after an otherwise successful call.
                options["strict"] = True
            structured_model = provider_model.model.with_structured_output(
                output_schema,
                method=self._structured_output_method,
                **options,
            )
        except Exception:
            raise _LLMInvocationFailure(
                reason="provider request failed",
                # A provider that cannot configure the requested schema should
                # not block another provider that supports it.
                fallback_allowed=True,
                provider_unavailable=False,
            ) from None

        config = {
            "metadata": metadata.as_dict(),
            "tags": [
                "deep-research-agent",
                role.value,
                metadata.node,
                provider_model.provider,
            ],
        }

        for attempt in range(1, self._structured_output_attempts + 1):
            try:
                result = await structured_model.ainvoke(
                    messages,
                    config=config,
                )

                if isinstance(result, output_schema):
                    return result

                if isinstance(result, dict):
                    return output_schema.model_validate(result)

                raise TypeError(
                    f"Unexpected structured output type: {type(result).__name__}"
                )

            except Exception as exc:
                should_retry = (
                    attempt < self._structured_output_attempts
                    and _is_retryable_output_error(exc)
                )

                if not should_retry:
                    structured_failure = _is_retryable_output_error(exc)
                    reason = (
                        "invalid structured output"
                        if structured_failure
                        else "provider request failed"
                    )
                    fallback_allowed = _provider_fallback_allowed(exc)
                    logger.warning(
                        "LLM provider invocation failed.",
                        extra={
                            "event": "llm_provider_failed",
                            **metadata.as_dict(),
                            "role": role.value,
                            "schema": output_schema.__name__,
                            "provider": provider_model.provider,
                            "model": _model_name(provider_model.model),
                            "attempt": attempt,
                            "max_attempts": self._structured_output_attempts,
                            "failure_type": reason,
                            **_safe_provider_error_details(exc),
                        },
                    )
                    raise _LLMInvocationFailure(
                        reason=f"{reason} after {attempt} attempt(s)",
                        fallback_allowed=fallback_allowed,
                        provider_unavailable=(
                            fallback_allowed and not structured_failure
                        ),
                    ) from None

                logger.warning(
                    "Structured LLM output failed; retrying.",
                    extra={
                        "event": "structured_llm_retry",
                        **metadata.as_dict(),
                        "role": role.value,
                        "schema": output_schema.__name__,
                        "provider": provider_model.provider,
                        "attempt": attempt,
                        "max_attempts": self._structured_output_attempts,
                        "error_type": type(exc).__name__,
                    },
                )

                # Keep retries short to avoid unnecessary latency.
                await asyncio.sleep(0.5 * attempt)

        raise _LLMInvocationFailure(
            reason="structured invocation ended unexpectedly",
            fallback_allowed=True,
            provider_unavailable=False,
        )
