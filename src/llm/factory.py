"""Factory functions for constructing provider-independent LLM models."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from src.config import Settings
from src.llm.roles import LLMRole
from src.llm.service import LLMService


SUPPORTED_PROVIDERS = {"groq", "google_genai"}


def _resolve_model(
    *,
    model_name: str,
    default_provider: str,
) -> tuple[str, str]:
    """Resolve the provider and model name from application configuration."""

    model_name = model_name.strip()

    if ":" in model_name:
        provider, model_name = model_name.split(":", 1)
        provider = provider.strip().lower()
        model_name = model_name.strip()
    else:
        provider = default_provider

    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider '{provider}'. "
            f"Supported providers: {sorted(SUPPORTED_PROVIDERS)}"
        )

    if not model_name:
        raise ValueError("LLM model name cannot be empty.")

    return provider, model_name


def _create_chat_model(
    *,
    model_name: str,
    settings: Settings,
) -> BaseChatModel:
    """Create a chat model using its configured provider."""

    provider, resolved_model = _resolve_model(
        model_name=model_name,
        default_provider=settings.llm.llm_provider,
    )

    return init_chat_model(
        model=resolved_model,
        model_provider=provider,
        timeout=settings.llm.llm_timeout_seconds,
        max_retries=settings.llm.llm_max_retries,
    )


def create_llm_service(settings: Settings) -> LLMService:
    """Create a provider-independent LLM service for every research role."""

    role_models = {
        LLMRole.PLANNER: settings.llm.planner_model,
        LLMRole.QUERY_GENERATOR: settings.llm.query_generator_model,
        LLMRole.EVIDENCE_EXTRACTOR: settings.llm.evidence_model,
        LLMRole.COVERAGE_EVALUATOR: settings.llm.coverage_model,
        LLMRole.WRITER: settings.llm.writer_model,
        LLMRole.CITATION_CHECKER: settings.llm.citation_model,
    }

    models = {
        role: _create_chat_model(
            model_name=model_name,
            settings=settings,
        )
        for role, model_name in role_models.items()
    }

    return LLMService(
        models=models,
        structured_output_method=settings.llm.llm_structured_output_method,
        structured_output_attempts=settings.llm.llm_structured_output_attempts,
    )

