"""Focused offline tests for ordered LLM provider fallback."""

import asyncio
from types import SimpleNamespace

import pytest
from langchain_core.exceptions import OutputParserException

from src.llm import (
    LLMCallMetadata,
    LLMProviderModel,
    LLMRole,
    LLMService,
    LLMServiceError,
)
from src.llm import factory as llm_factory
from src.models import EvidenceExtractionOutput, SemanticCitationCheckOutput


class FakeModel:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0
        self.structured_options = None

    def with_structured_output(self, output_schema, method, **options):
        self.structured_options = options
        return self

    async def ainvoke(self, messages, config):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _metadata(*, job_id: str = "job-1", private: bool = False) -> LLMCallMetadata:
    return LLMCallMetadata(
        job_id=job_id,
        research_session_id="session-1",
        node="test",
        contains_private_data=private,
    )


def _service(
    *models: tuple[str, FakeModel],
    attempts: int = 1,
    private_providers: set[str] | None = None,
) -> LLMService:
    return LLMService(
        {
            LLMRole.EVIDENCE_EXTRACTOR: [
                LLMProviderModel(provider=provider, model=model)
                for provider, model in models
            ]
        },
        structured_output_attempts=attempts,
        private_data_providers=private_providers,
    )


async def _invoke(service: LLMService, metadata: LLMCallMetadata):
    return await service.invoke_structured(
        role=LLMRole.EVIDENCE_EXTRACTOR,
        messages=[],
        output_schema=EvidenceExtractionOutput,
        metadata=metadata,
    )


def test_unavailable_provider_falls_back_and_healthy_provider_stays_pinned():
    openai = FakeModel(RuntimeError("service unavailable"))
    gemini = FakeModel(EvidenceExtractionOutput(evidence=[]))
    groq = FakeModel(EvidenceExtractionOutput(evidence=[]))
    service = _service(
        ("openai", openai),
        ("google_genai", gemini),
        ("groq", groq),
    )

    async def run():
        await _invoke(service, _metadata())
        await _invoke(service, _metadata())

    asyncio.run(run())

    assert openai.calls == 1
    assert gemini.calls == 2
    assert groq.calls == 0

    service.release_job("job-1")
    asyncio.run(_invoke(service, _metadata()))
    assert openai.calls == 2


def test_repeated_structured_output_failure_moves_to_next_provider():
    openai = FakeModel(OutputParserException("private malformed output"))
    gemini = FakeModel(EvidenceExtractionOutput(evidence=[]))
    service = _service(
        ("openai", openai),
        ("google_genai", gemini),
        attempts=2,
    )

    result = asyncio.run(_invoke(service, _metadata()))

    assert result.evidence == []
    assert openai.calls == 2
    assert gemini.calls == 1


def test_structured_output_failure_does_not_blacklist_the_only_provider():
    groq = FakeModel(OutputParserException("malformed output"))
    service = _service(("groq", groq))

    with pytest.raises(LLMServiceError, match="invalid structured output"):
        asyncio.run(_invoke(service, _metadata()))

    groq.outcome = EvidenceExtractionOutput(evidence=[])
    result = asyncio.run(_invoke(service, _metadata()))

    assert result.evidence == []
    assert groq.calls == 2


def test_groq_tool_call_failure_is_retried_as_structured_output_error():
    groq = FakeModel(RuntimeError("tool_use_failed: failed to call a function"))
    service = _service(("groq", groq), attempts=2)

    with pytest.raises(LLMServiceError, match="invalid structured output"):
        asyncio.run(_invoke(service, _metadata()))

    assert groq.calls == 2


def test_groq_json_validation_failure_is_retried_from_error_code():
    class GroqValidationError(RuntimeError):
        body = {
            "error": {
                "code": "json_validate_failed",
                "type": "invalid_request_error",
                "failed_generation": "PRIVATE GENERATED CONTENT",
            }
        }

    groq = FakeModel(GroqValidationError("Bad request"))
    service = _service(("groq", groq), attempts=2)

    with pytest.raises(LLMServiceError, match="invalid structured output"):
        asyncio.run(_invoke(service, _metadata()))

    assert groq.calls == 2


def test_groq_json_schema_uses_strict_structured_output():
    groq = FakeModel(SemanticCitationCheckOutput(
        revised_executive_summary="Supported summary.",
        finding_results=[],
    ))
    service = LLMService({
        LLMRole.CITATION_CHECKER: [
            LLMProviderModel(provider="groq", model=groq),
        ],
    })

    asyncio.run(service.invoke_structured(
        role=LLMRole.CITATION_CHECKER,
        messages=[],
        output_schema=SemanticCitationCheckOutput,
        metadata=_metadata(),
    ))

    assert groq.structured_options == {"strict": True}


def test_groq_evidence_extraction_uses_strict_structured_output():
    groq = FakeModel(EvidenceExtractionOutput(evidence=[]))
    service = _service(("groq", groq))

    asyncio.run(_invoke(service, _metadata()))

    assert groq.structured_options == {"strict": True}


def test_terminal_context_error_does_not_fall_back():
    openai = FakeModel(RuntimeError("context length exceeded"))
    gemini = FakeModel(EvidenceExtractionOutput(evidence=[]))
    service = _service(("openai", openai), ("google_genai", gemini))

    with pytest.raises(LLMServiceError, match="provider request failed"):
        asyncio.run(_invoke(service, _metadata()))

    assert gemini.calls == 0


def test_private_data_uses_only_allowlisted_providers():
    openai = FakeModel(EvidenceExtractionOutput(evidence=[]))
    gemini = FakeModel(EvidenceExtractionOutput(evidence=[]))
    groq = FakeModel(EvidenceExtractionOutput(evidence=[]))
    service = _service(
        ("openai", openai),
        ("google_genai", gemini),
        ("groq", groq),
        private_providers={"groq"},
    )

    asyncio.run(_invoke(service, _metadata(private=True)))

    assert openai.calls == 0
    assert gemini.calls == 0
    assert groq.calls == 1


def test_model_chain_skips_provider_that_cannot_initialize(monkeypatch):
    models = {
        "google_genai": FakeModel(EvidenceExtractionOutput(evidence=[])),
        "groq": FakeModel(EvidenceExtractionOutput(evidence=[])),
    }

    def create_model(*, model_name, settings):
        provider = model_name.split(":", 1)[0]
        if provider == "openai":
            raise ImportError("missing integration")
        return models[provider]

    monkeypatch.setattr(llm_factory, "_create_chat_model", create_model)
    settings = SimpleNamespace(llm=SimpleNamespace(llm_provider="groq"))

    chain = llm_factory._create_model_chain(
        model_names="openai:o,google_genai:g,groq:r",
        settings=settings,
        provider_order=["openai", "google_genai", "groq"],
    )

    assert [entry.provider for entry in chain] == ["google_genai", "groq"]


def test_model_not_found_falls_back_and_logs_the_configured_model(caplog):
    class ModelNotFound(Exception):
        status_code = 404
        body = {"error": {"code": "model_not_found", "type": "invalid_request_error"}}

    openai = FakeModel(ModelNotFound("The model does not exist"))
    openai.model_name = "gpt-typo"
    groq = FakeModel(EvidenceExtractionOutput(evidence=[]))
    service = _service(("openai", openai), ("groq", groq))

    result = asyncio.run(_invoke(service, _metadata()))

    assert result == EvidenceExtractionOutput(evidence=[])
    assert groq.calls == 1
    failure = next(record for record in caplog.records if getattr(record, "event", "") == "llm_provider_failed")
    assert failure.model == "gpt-typo"
