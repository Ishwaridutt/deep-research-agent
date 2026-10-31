"""Offline tests for the Jev client: request format, answer parsing, retries, and limits."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr

from src.llm import JevService, JevServiceError, create_jev_service


def service(handler, **overrides):
    options = {
        "api_key": "test-key", "model": "jev-1.14.0", "api_url": "https://jev.test/v1/systemone",
        "timeout_seconds": 5, "max_concurrency": 4, "max_retries": 2, "retry_delay_seconds": 0,
        "client": httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    }
    return JevService(**{**options, **overrides})


def answer(body):
    return httpx.Response(200, json={"model": "jev-1.14.0", "answers": {"answer": body},
                                     "usage": {"input_tokens": 42, "output_tokens": 1}})


def test_noul_sends_typed_question_with_bearer_key_and_returns_probability():
    requests = []

    def handler(request):
        requests.append(request)
        return answer({"type": "noul", "noul": 0.83})

    probability = asyncio.run(service(handler).noul(
        state={"passage": "text"}, instructions="Is it useful?", criteria={"true": "yes", "false": "no"},
    ))
    assert probability == 0.83
    request = requests[0]
    assert request.url == "https://jev.test/v1/systemone"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {
        "model": "jev-1.14.0", "state": {"passage": "text"},
        "questions": {"answer": {"type": "noul", "instructions": "Is it useful?",
                                 "criteria": {"true": "yes", "false": "no"}}},
    }


def test_choice_returns_top_choice_confidence_and_probabilities():
    def handler(request):
        return answer({"type": "choice", "choice": "b", "confidence": 0.9, "probabilities": {"a": 0.1, "b": 0.9}})

    result = asyncio.run(service(handler).choice(state="x", instructions="Which?", criteria={"a": "A", "b": "B"}))
    assert result == ("b", 0.9, {"a": 0.1, "b": 0.9})


def test_unknown_choice_or_malformed_answer_raises_service_error():
    def unknown(request):
        return answer({"type": "choice", "choice": "z", "confidence": 1, "probabilities": {"z": 1}})

    with pytest.raises(JevServiceError):
        asyncio.run(service(unknown).choice(state="x", instructions="Which?", criteria={"a": "A"}))
    with pytest.raises(JevServiceError):
        asyncio.run(service(lambda request: answer({"type": "noul"})).noul(state="x", instructions="?"))
    with pytest.raises(JevServiceError):
        asyncio.run(service(lambda request: httpx.Response(200, json={"answers": {}})).noul(state="x", instructions="?"))


def test_rate_limit_and_server_errors_are_retried_then_succeed():
    statuses = iter([429, 503, 200])
    calls = []

    def handler(request):
        calls.append(request)
        status = next(statuses)
        return answer({"type": "noul", "noul": 0.4}) if status == 200 else httpx.Response(status)

    assert asyncio.run(service(handler).noul(state="x", instructions="?")) == 0.4
    assert len(calls) == 3


def test_client_errors_are_not_retried_and_exhausted_retries_raise():
    calls = []

    def bad_request(request):
        calls.append(request)
        return httpx.Response(400, json={"detail": "bad"})

    with pytest.raises(JevServiceError, match="HTTP 400"):
        asyncio.run(service(bad_request).noul(state="x", instructions="?"))
    assert len(calls) == 1

    calls.clear()
    with pytest.raises(JevServiceError, match="HTTP 429"):
        asyncio.run(service(lambda request: calls.append(request) or httpx.Response(429)).noul(state="x", instructions="?"))
    assert len(calls) == 3


def test_timeouts_become_service_errors():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(JevServiceError, match="ReadTimeout"):
        asyncio.run(service(handler, max_retries=0).noul(state="x", instructions="?"))


def test_concurrent_requests_stay_within_the_limit():
    active = 0
    peak = 0

    async def handler(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return answer({"type": "noul", "noul": 0.5})

    async def run():
        jev = service(handler, max_concurrency=2)
        return await asyncio.gather(*(jev.noul(state=index, instructions="?") for index in range(6)))

    assert asyncio.run(run()) == [0.5] * 6
    assert peak == 2


def jev_settings(**llm):
    defaults = {
        "typesafe_api_key": SecretStr("key"), "jev_model": "jev-1.14.0", "jev_api_url": "https://jev.test",
        "jev_timeout_seconds": 5, "jev_max_concurrency": 8, "jev_max_retries": 2, "jev_retry_delay_seconds": 1,
        "guardrail_use_jev": True, "evidence_use_jev": True, "llm_private_data_providers": "openai,groq",
    }
    return SimpleNamespace(llm=SimpleNamespace(**{**defaults, **llm}))


def test_factory_needs_a_key_and_reads_private_data_permission():
    assert create_jev_service(jev_settings(typesafe_api_key=None)) is None
    assert create_jev_service(jev_settings(guardrail_use_jev=False, evidence_use_jev=False)) is None
    assert create_jev_service(jev_settings()).private_data_allowed is False
    assert create_jev_service(jev_settings(llm_private_data_providers="openai,typesafe")).private_data_allowed is True
