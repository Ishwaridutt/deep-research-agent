"""Factory functions for constructing provider-independent LLM models."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from src.config import Settings
from src.llm.roles import LLMRole
from src.llm.service import LLMService


def _create_chat_model(
    *,
    model_name: str,
    settings: Settings,
) -> BaseChatModel:
    """
    Creates a chat model from a provider-prefixed model name.

    Examples:
        groq:some-model
        openai:some-model
        google_genai:some-model
        xai:some-model

    LangChain resolves the appropriate provider integration at runtime.
    """

    return init_chat_model(
        model=model_name,
        model_provider="groq",
        timeout=settings.llm.llm_timeout_seconds,
        max_retries=settings.llm.llm_max_retries,
    )


def create_llm_service(
    settings: Settings,
) -> LLMService:
    """
    Creates the application's provider-independent LLM service.

    Each role can use a different model or even a different provider,
    entirely through configuration.
    """

    models: dict[LLMRole, BaseChatModel] = {
        LLMRole.PLANNER: _create_chat_model(
            model_name=settings.llm.planner_model,
            settings=settings,
        ),
        LLMRole.QUERY_GENERATOR: _create_chat_model(
            model_name=settings.llm.query_generator_model,
            settings=settings,
        ),
        LLMRole.EVIDENCE_EXTRACTOR: _create_chat_model(
            model_name=settings.llm.evidence_model,
            settings=settings,
        ),
        LLMRole.COVERAGE_EVALUATOR: _create_chat_model(
            model_name=settings.llm.coverage_model,
            settings=settings,
        ),
        LLMRole.WRITER: _create_chat_model(
            model_name=settings.llm.writer_model,
            settings=settings,
        ),
        LLMRole.CITATION_CHECKER: _create_chat_model(
            model_name=settings.llm.citation_model,
            settings=settings,
        ),
    }

    return LLMService(models=models)

