"""Focused offline tests for Phase 3 source selection and graph integration."""

import asyncio
from types import SimpleNamespace

import pytest

from src.agents.research.research_source_selection_node import (
    research_source_selection_node,
)
from src.graph.graph_routing import route_after_source_selection
from src.graph.graph_state import create_initial_workflow_state
from src.guardrails.input_guardrail import input_guardrail_node
from src.models import (
    ResearchExecutionStatus,
    ResearchRequest,
    ResearchSource,
    ResearchSourceMode,
    ResearchSourceSelection,
)
from src.services.research_job_resources import ResearchJobResources
from src.services.research_service import ResearchService


class FakeSession:
    """Track cleanup without constructing embedding dependencies."""

    def __init__(self) -> None:
        self.clear_calls = 0

    def clear(self) -> None:
        self.clear_calls += 1


class FakeSessionFactory:
    """Create deterministic private-document sessions."""

    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []

    def create_session(self) -> FakeSession:
        session = FakeSession()
        self.sessions.append(session)
        return session


class FakeDocumentService:
    """Capture the document preparation boundary."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, list[str], FakeSession]] = []

    async def build_document_retrieval_session(
        self,
        owner_id: str,
        document_ids: list[str],
        session: FakeSession,
    ) -> None:
        self.calls.append((owner_id, document_ids, session))
        if self.error is not None:
            raise self.error


def create_document_state(*, hybrid: bool = False):
    """Create the minimum complete workflow state for source-selection tests."""

    allowed_sources = [ResearchSource.DOCUMENTS]
    mode = ResearchSourceMode.DOCUMENTS
    if hybrid:
        allowed_sources.insert(0, ResearchSource.WEB)
        mode = ResearchSourceMode.HYBRID

    return create_initial_workflow_state(
        job_id="job-1",
        research_session_id="session-1",
        question="Summarize the selected documents",
        allowed_sources=allowed_sources,
        source_mode=mode,
        owner_id="owner-1",
        selected_document_ids=["doc-1", "doc-2"],
    )


def test_source_selection_builds_one_session_and_reuses_it():
    """Prepare private documents once across repeated graph iterations."""

    async def run() -> None:
        state = create_document_state()
        factory = FakeSessionFactory()
        document_service = FakeDocumentService()
        resources = ResearchJobResources()
        config = {"configurable": {"job_resources": resources}}

        await research_source_selection_node(
            state,
            config,
            document_service=document_service,
            retrieval_session_factory=factory,
        )
        await research_source_selection_node(
            state,
            config,
            document_service=document_service,
            retrieval_session_factory=factory,
        )

        assert len(factory.sessions) == 1
        assert document_service.calls == [
            ("owner-1", ["doc-1", "doc-2"], factory.sessions[0])
        ]
        assert resources.document_retrieval_session is factory.sessions[0]

    asyncio.run(run())


def test_web_source_selection_does_not_create_document_resources():
    """Leave private-document dependencies untouched for web-only research."""

    async def run() -> None:
        state = create_initial_workflow_state(
            job_id="job-1",
            research_session_id="session-1",
            question="What changed on the web?",
            allowed_sources=[ResearchSource.WEB],
        )
        factory = FakeSessionFactory()
        document_service = FakeDocumentService()

        result = await research_source_selection_node(
            state,
            {},
            document_service=document_service,
            retrieval_session_factory=factory,
        )

        assert result == {
            "source_mode": ResearchSourceMode.WEB,
            "selected_document_ids": [],
        }
        assert factory.sessions == []
        assert document_service.calls == []

    asyncio.run(run())


def test_source_selection_records_authorization_failure_and_clears():
    """Convert an owner-scoped lookup failure into a terminal graph error."""

    async def run() -> None:
        state = create_document_state()
        factory = FakeSessionFactory()
        document_service = FakeDocumentService(ValueError("Document not found."))
        resources = ResearchJobResources()

        result = await research_source_selection_node(
            state,
            {"configurable": {"job_resources": resources}},
            document_service=document_service,
            retrieval_session_factory=factory,
        )

        assert resources.document_retrieval_session is None
        assert factory.sessions[0].clear_calls == 1
        assert result["errors"][-1].recoverable is False
        assert result["errors"][-1].message == "Document not found."

    asyncio.run(run())


def test_source_selection_routes_web_document_and_hybrid_modes():
    """Keep web and document branches distinct at the graph boundary."""

    document_state = create_document_state()
    hybrid_state = create_document_state(hybrid=True)

    assert route_after_source_selection(document_state) == "document_research"
    assert route_after_source_selection(hybrid_state) == "web_research"


def test_research_request_defaults_to_web_and_validates_document_selection():
    """Expose one explicit source-selection contract to API callers."""

    request = ResearchRequest(question="What changed?")
    assert request.source_selection.mode == ResearchSourceMode.WEB

    with pytest.raises(ValueError, match="requires at least one"):
        ResearchSourceSelection(mode=ResearchSourceMode.DOCUMENTS)


def test_input_guardrail_accepts_valid_document_selection():
    """Allow private documents once identity and document IDs are present."""

    state = create_document_state()
    result = input_guardrail_node(
        state,
        settings=SimpleNamespace(
            research_limits=SimpleNamespace(max_question_chars=1000)
        ),
    )

    assert result["allowed_sources"] == [ResearchSource.DOCUMENTS]
    assert "errors" not in result


def test_research_service_clears_runtime_resources_when_graph_fails():
    """Guarantee session cleanup on the synchronous exceptional path."""

    async def run() -> None:
        session = FakeSession()

        class FailingGraph:
            async def ainvoke(self, state, config):
                resources = config["configurable"]["job_resources"]
                resources.document_retrieval_session = session
                raise RuntimeError("graph failed")

        service = ResearchService(
            graph=FailingGraph(),
            settings=SimpleNamespace(
                app=SimpleNamespace(graph_recursion_limit=10)
            ),
            job_store=SimpleNamespace(),
        )

        result = await service.research(question="What changed?")

        assert result.status == ResearchExecutionStatus.FAILED
        assert session.clear_calls == 1

    asyncio.run(run())


def test_persisted_job_clears_runtime_resources_when_cancelled(monkeypatch):
    """Guarantee private-session cleanup during server-shutdown cancellation."""

    async def run() -> None:
        session = FakeSession()

        async def cancelled_run(*, config, **kwargs):
            resources = config["configurable"]["job_resources"]
            resources.document_retrieval_session = session
            raise asyncio.CancelledError

        class FakeJobStore:
            async def mark_running(self, job_id):
                return None

            async def append_event(self, event):
                return None

            async def finish_job(self, **kwargs):
                return None

        monkeypatch.setattr(
            "src.services.research_service.run_graph_with_events",
            cancelled_run,
        )
        service = ResearchService(
            graph=SimpleNamespace(),
            settings=SimpleNamespace(
                app=SimpleNamespace(graph_recursion_limit=10)
            ),
            job_store=FakeJobStore(),
        )
        state = service._create_state(
            question="What changed?",
            source_selection=None,
            owner_id=None,
            research_session_id=None,
        )

        with pytest.raises(asyncio.CancelledError):
            await service._run_persisted_job(state)

        assert session.clear_calls == 1

    asyncio.run(run())
