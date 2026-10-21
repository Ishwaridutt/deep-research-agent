"""Factory functions for constructing provider-independent LLM models."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from src.config import Settings
from src.llm.roles import LLMRole
from src.llm.service import LLMProviderModel, LLMService


SUPPORTED_PROVIDERS = {"openai", "google_genai", "groq"}


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


def _provider_names(value: str, *, setting_name: str) -> list[str]:
    providers = list(dict.fromkeys(
        provider.strip().lower() for provider in value.split(",") if provider.strip()
    ))
    unsupported = set(providers) - SUPPORTED_PROVIDERS
    if not providers or unsupported:
        raise ValueError(
            f"{setting_name} must contain only: {sorted(SUPPORTED_PROVIDERS)}"
        )
    return providers


def _create_model_chain(
    *,
    model_names: str,
    settings: Settings,
    provider_order: list[str],
) -> list[LLMProviderModel]:
    """Build one role chain and order its provider-qualified models."""

    resolved = [
        _resolve_model(
            model_name=model_name,
            default_provider=settings.llm.llm_provider,
        )
        for model_name in model_names.split(",")
        if model_name.strip()
    ]
    by_provider = dict(resolved)
    if len(by_provider) != len(resolved):
        raise ValueError("Each LLM role may configure only one model per provider.")
    missing_from_order = set(by_provider) - set(provider_order)
    if missing_from_order:
        raise ValueError(
            "Every configured model provider must appear in LLM_PROVIDER_ORDER."
        )

    models: list[LLMProviderModel] = []
    for provider in provider_order:
        if provider not in by_provider:
            continue
        try:
            model = _create_chat_model(
                model_name=f"{provider}:{by_provider[provider]}",
                settings=settings,
            )
        except Exception:
            # An absent SDK or credential disables only that provider at startup.
            print("LLM provider unavailable during setup:", f"provider={provider}")
            continue
        models.append(LLMProviderModel(provider=provider, model=model))

    if not models:
        raise ValueError("No configured LLM provider could be initialized.")
    return models


def create_llm_service(settings: Settings) -> LLMService:
    """Create a provider-independent LLM service for every research role."""

    role_models = {
        LLMRole.PLANNER: settings.llm.planner_model,
        LLMRole.QUERY_GENERATOR: settings.llm.query_generator_model,
        LLMRole.EVIDENCE_EXTRACTOR: settings.llm.evidence_model,
        LLMRole.COVERAGE_EVALUATOR: settings.llm.coverage_model,
        LLMRole.WRITER: settings.llm.writer_model,
        LLMRole.CITATION_CHECKER: settings.llm.citation_model,
        LLMRole.CRITIC: settings.llm.critic_model or settings.llm.citation_model,
        LLMRole.INPUT_GUARDRAIL: settings.llm.guardrail_model or settings.llm.query_generator_model,
    }

    provider_order = _provider_names(
        settings.llm.llm_provider_order,
        setting_name="LLM_PROVIDER_ORDER",
    )
    private_data_providers = set(_provider_names(
        settings.llm.llm_private_data_providers,
        setting_name="LLM_PRIVATE_DATA_PROVIDERS",
    ))
    models = {
        role: _create_model_chain(
            model_names=model_names,
            settings=settings,
            provider_order=provider_order,
        )
        for role, model_names in role_models.items()
    }

    return LLMService(
        models=models,
        structured_output_method=settings.llm.llm_structured_output_method,
        structured_output_attempts=settings.llm.llm_structured_output_attempts,
        private_data_providers=private_data_providers,
    )

