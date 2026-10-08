"""Shared service for invoking provider-independent LLMs."""

from dataclasses import dataclass
from typing import Sequence, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from src.llm.roles import LLMRole


StructuredOutput = TypeVar(
    "StructuredOutput",
    bound=BaseModel,
)


@dataclass(slots=True)
class LLMCallMetadata:
    """
    Metadata attached to an LLM call for tracing and observability.
    """

    job_id: str
    research_session_id: str
    node: str
    iteration: int | None = None

    def as_dict(self) -> dict[str, str | int]:
        """Converts metadata into a LangChain-compatible dictionary."""

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

    def __init__(
        self,
        *,
        role: LLMRole,
        message: str,
    ) -> None:
        """Creates an error associated with a logical LLM role."""

        self.role = role
        super().__init__(message)


class LLMService:
    """
    Provider-independent interface used by all LLM-backed nodes.

    The rest of the application never needs to know whether a model is
    hosted by Groq, OpenAI, Google, xAI, or another supported provider.
    """

    def __init__(
        self,
        models: dict[LLMRole, BaseChatModel],
    ) -> None:
        """Stores configured chat models by logical application role."""

        self._models = models

    def _get_model(
        self,
        role: LLMRole,
    ) -> BaseChatModel:
        """Returns the model configured for a logical LLM role."""

        model = self._models.get(role)

        if model is None:
            raise LLMServiceError(
                role=role,
                message=(
                    f"No LLM model configured for role "
                    f"'{role.value}'."
                ),
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
        """
        Invokes the selected model and returns validated structured output.
        """

        model = self._get_model(role)

        structured_model = model.with_structured_output(
            output_schema,
        )

        try:
            result = await structured_model.ainvoke(
                messages,
                config={
                    "metadata": metadata.as_dict(),
                    "tags": [
                        "deep-research-agent",
                        role.value,
                        metadata.node,
                    ],
                },
            )

        except Exception as exc:
            raise LLMServiceError(
                role=role,
                message=(
                    f"LLM invocation failed for role "
                    f"'{role.value}': {exc}"
                ),
            ) from exc

        if not isinstance(result, output_schema):
            raise LLMServiceError(
                role=role,
                message=(
                    f"LLM role '{role.value}' returned an "
                    f"unexpected output type."
                ),
            )

        return result

