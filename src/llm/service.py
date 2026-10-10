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

    def as_dict(self) -> dict[str, str | int]:
        """Convert metadata into a LangChain-compatible dictionary."""

        metadata: dict[str, str | int] = {
            "job_id": self.job_id,
            "research_session_id": self.research_session_id,
            "node": self.node,
        }

        if self.iteration is not None:
            metadata["iteration"] = self.iteration

        return metadata


class LLMServiceError(RuntimeError):
    """Raised when an LLM invocation fails."""

    def __init__(self, *, role: LLMRole, message: str) -> None:
        self.role = role
        super().__init__(message)


def _is_retryable_output_error(exc: Exception) -> bool:
    """Retry malformed structured output, not provider or quota failures."""

    if isinstance(exc, (OutputParserException, ValidationError, json.JSONDecodeError)):
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
        "generated json does not match the expected schema",
        "invalid json",
        "json decode",
        "could not parse",
        "output parsing",
    )

    return any(reason in message for reason in retryable)


class LLMService:
    """Provider-independent interface used by all LLM-backed nodes."""

    def __init__(
        self,
        models: dict[LLMRole, BaseChatModel],
        *,
        structured_output_method: StructuredOutputMethod = "json_schema",
        structured_output_attempts: int = 2,
    ) -> None:
        self._models = models
        self._structured_output_method = structured_output_method
        self._structured_output_attempts = max(1, structured_output_attempts)

    def _get_model(self, role: LLMRole) -> BaseChatModel:
        """Return the model configured for an application role."""

        model = self._models.get(role)

        if model is None:
            raise LLMServiceError(
                role=role,
                message=f"No LLM model configured for role '{role.value}'.",
            )

        return model

    async def invoke_structured(
        self,
        *,
        role: LLMRole,
        messages: Sequence[BaseMessage],
        output_schema: type[StructuredOutput],
        metadata: LLMCallMetadata,
    ) -> StructuredOutput:
        """Invoke an LLM with schema validation and bounded parsing retries."""

        model = self._get_model(role)

        try:
            structured_model = model.with_structured_output(
                output_schema,
                method=self._structured_output_method,
            )
        except Exception as exc:
            raise LLMServiceError(
                role=role,
                message=f"Could not configure structured output for '{role.value}': {exc}",
            ) from exc

        config = {
            "metadata": metadata.as_dict(),
            "tags": [
                "deep-research-agent",
                role.value,
                metadata.node,
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
                    raise LLMServiceError(
                        role=role,
                        message=(
                            f"LLM invocation failed for role '{role.value}' "
                            f"after {attempt} attempt(s): {exc}"
                        ),
                    ) from exc

                logger.warning(
                    "Structured LLM output failed; retrying.",
                    extra={
                        "event": "structured_llm_retry",
                        **metadata.as_dict(),
                        "role": role.value,
                        "schema": output_schema.__name__,
                        "attempt": attempt,
                        "max_attempts": self._structured_output_attempts,
                        "error_type": type(exc).__name__,
                    },
                )

                # Keep retries short to avoid unnecessary latency.
                await asyncio.sleep(0.5 * attempt)

        raise RuntimeError("Structured LLM invocation ended unexpectedly.")

    