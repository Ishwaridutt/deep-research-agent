"""Tests for single-run LangGraph event recording and SSE serialization."""

import asyncio
from pathlib import Path

from src.models import ErrorType, ResearchError
from src.services.research_pipeline_stream_service import (
    PipelineRecorder,
    _safe_details,
    encode_sse,
    run_graph_with_events,
)


def test_source_progress_details_expose_counts_not_private_data():
    selection = _safe_details(
        "research_source_selection",
        {
            "source_mode": "hybrid",
            "selected_document_ids": ["private-1", "private-2"],
        },
    )
    retrieval = _safe_details(
        "research_retrieval",
        {
            "research": {
                "web_retrieval_results": [
                    {"chunks": [{"content": "SECRET"}]}
                ],
                "document_retrieval_results": [
                    {"chunks": [{"filename": "private.pdf"}, {}]}
                ],
            }
        },
    )

    assert selection == {"source_mode": "hybrid", "document_count": 2}
    assert retrieval == {
        "web_result_count": 1,
        "document_result_count": 1,
        "web_chunk_count": 1,
        "document_chunk_count": 2,
    }
    assert "SECRET" not in str(retrieval)
    assert "private.pdf" not in str(retrieval)


def test_provider_progress_exposes_only_safe_counts_and_branch_names():
    research = {
        "search_queries": [{"id": "query-1", "iteration": 1}],
        "search_results": [{
            "id": "result-1",
            "query_id": "query-1",
            "provider": "brave",
            "url": "https://private.example/secret",
            "snippet": "SECRET",
            "provider_metadata": {
                "search_diagnostics": {
                    "successful_branches": ["web", "wikipedia"],
                    "failed_branches": ["arxiv"],
                    "web_fallback_activated": True,
                },
            },
        }],
        "sources": [{
            "provider": "brave",
            "search_result_ids": ["result-1"],
            "content": "SECRET SOURCE BODY",
            "provider_metadata": {"extraction_provider": "firecrawl"},
        }],
    }

    search = _safe_details(
        "research_web_search",
        {"research": research},
        iteration=1,
    )
    extraction = _safe_details(
        "research_source_processor",
        {"research": research},
        iteration=1,
    )

    assert search == {
        "result_count": 1,
        "provider_counts": {"brave": 1},
        "failed_branches": ["arxiv"],
        "partial_failure_count": 1,
        "web_fallback_activated": True,
    }
    assert extraction == {
        "source_count": 1,
        "provider_counts": {"brave": 1},
        "extraction_fallback_count": 1,
    }
    assert "SECRET" not in str(search)
    assert "SECRET" not in str(extraction)
    assert "private.example" not in str(search)


def test_ui_polls_status_only_as_an_sse_fallback():
    script = (
        Path(__file__).parents[1] / "static" / "js" / "app.js"
    ).read_text(encoding="utf-8")

    assert "function startPolling" not in script
    assert "pollTimer = setTimeout" in script
    assert "pollDelay = Math.min(pollDelay * 2, FALLBACK_POLL_MAX_MS)" in script
    assert "streamAvailable = true;\n        stopFallbackPolling();" in script
    assert "void checkStatusAfterStreamFailure(token);" in script


def test_ui_marks_the_running_pipeline_stage_failed():
    script = (
        Path(__file__).parents[1] / "static" / "js" / "app.js"
    ).read_text(encoding="utf-8")

    assert 'if (event.event_type === "workflow_failed") pipelineFailed = true;' in script
    assert 'return state === "running" && pipelineFailed ? "failed" : state;' in script
    assert 'pipelineFailed = final.status !== "completed" || !final.report;' in script
    assert 'failed: "×"' in script


def test_ui_surfaces_missing_document_selection():
    script = (
        Path(__file__).parents[1] / "static" / "js" / "app.js"
    ).read_text(encoding="utf-8")
    styles = (
        Path(__file__).parents[1] / "static" / "css" / "styles.css"
    ).read_text(encoding="utf-8")

    assert "Select at least one document before starting" in script
    assert 'setDocumentStatus(message, "error", DOCUMENT_SELECTION_ISSUE);' in script
    assert 'type === "error" ? "alert" : "status"' in script
    assert "if (!usesDocuments(mode)) clearDocumentSelectionError();" in script
    assert "replayAttention(ui.documentPanel);" in script
    assert "replayAttention(ui.status);" in script
    assert "async function validateSelectedDocuments()" in script
    assert "!await validateSelectedDocuments()" in script
    assert "One or more selected documents are no longer available" in script
    assert "error.status = response.status;" in script
    assert "usesDocuments(mode) && error?.status === 422" in script
    assert 'showDocumentIssue("Documents unavailable", message);' in script
    assert '.status-panel[data-issue="document-selection"] { position: fixed;' in styles
    assert "animation: validation-toast" in styles


class FakeGraph:
    """Return LangGraph-style v2 events without calling a real LLM."""

    def __init__(self, final_state):
        self.final_state = final_state
        self.calls = 0

    async def astream_events(self, initial_state, *, config, version):
        assert version == "v2"
        assert config == {"recursion_limit": 50}

        self.calls += 1

        yield {
            "event": "on_chain_start",
            "run_id": "root",
            "parent_ids": [],
            "name": "deep_research_job",
            "data": {"input": initial_state},
        }

        yield {
            "event": "on_chain_start",
            "run_id": "planner",
            "parent_ids": ["root"],
            "name": "research_planner",
            "data": {},
        }

        yield {
            "event": "on_chain_end",
            "run_id": "planner",
            "parent_ids": ["root"],
            "name": "research_planner",
            "data": {
                "output": {
                    "private_text": "SECRET",
                }
            },
        }

        yield {
            "event": "on_chain_start",
            "run_id": "router",
            "parent_ids": ["root"],
            "name": "research_routing",
            "data": {},
        }

        yield {
            "event": "on_chain_end",
            "run_id": "router",
            "parent_ids": ["root"],
            "name": "research_routing",
            "data": {
                "output": {
                    "research_control": {
                        "last_decision": {
                            "action": "continue_research",
                            "reason": "A gap remains.",
                        },
                        "pending_gaps": ["gap"],
                    },
                },
            },
        }

        yield {
            "event": "on_chain_end",
            "run_id": "root",
            "parent_ids": [],
            "name": "deep_research_job",
            "data": {"output": self.final_state},
        }


def test_stream_captures_final_state_without_duplicate_execution():
    async def run():
        state = {
            "job_id": "job-1",
            "research": {
                "iteration_count": 0,
            },
        }

        graph = FakeGraph(state)
        events = []

        async def persist(event):
            events.append(event)

        recorder = PipelineRecorder(
            job_id="job-1",
            persist=persist,
        )

        final_state = await run_graph_with_events(
            graph=graph,
            state=state,
            config={"recursion_limit": 50},
            recorder=recorder,
        )

        # The final state must come from the same graph execution.
        assert final_state is state
        assert graph.calls == 1

        assert [event.event_type for event in events] == [
            "workflow_started",
            "stage_started",
            "stage_completed",
            "stage_started",
            "stage_completed",
            "research_retry",
        ]

        # Events must have consecutive sequence numbers.
        assert [event.sequence for event in events] == list(range(1, 7))

        # The research router requested another iteration.
        assert events[-1].iteration == 1
        assert events[-1].details["next_iteration"] == 2
        assert events[-1].details["gap_count"] == 1

        # SSE progress must not expose private graph outputs.
        payload = "".join(encode_sse(event) for event in events)

        assert "SECRET" not in payload
        assert "id: 6\n" in payload
        assert "event: research_retry\n" in payload

    asyncio.run(run())


class FakeFailureGraph:
    """Return a node result containing a terminal graph-state error."""

    def __init__(self, final_state):
        self.final_state = final_state

    async def astream_events(self, initial_state, *, config, version):
        yield {
            "event": "on_chain_start",
            "run_id": "root",
            "parent_ids": [],
            "name": "deep_research_job",
            "data": {"input": initial_state},
        }
        yield {
            "event": "on_chain_start",
            "run_id": "source-selection",
            "parent_ids": ["root"],
            "name": "research_source_selection",
            "data": {},
        }
        yield {
            "event": "on_chain_end",
            "run_id": "source-selection",
            "parent_ids": ["root"],
            "name": "research_source_selection",
            "data": {"output": {"errors": self.final_state["errors"]}},
        }
        yield {
            "event": "on_chain_end",
            "run_id": "root",
            "parent_ids": [],
            "name": "deep_research_job",
            "data": {"output": self.final_state},
        }


def test_terminal_node_error_emits_failed_stage_with_safe_reason():
    async def run():
        error = ResearchError(
            error_type=ErrorType.VALIDATION,
            component="research_source_selection",
            message="Document not found.",
            recoverable=False,
        )
        state = {
            "job_id": "job-1",
            "research": {"iteration_count": 0},
            "errors": [error],
        }
        events = []

        async def persist(event):
            events.append(event)

        final_state = await run_graph_with_events(
            graph=FakeFailureGraph(state),
            state=state,
            config={"recursion_limit": 50},
            recorder=PipelineRecorder(job_id="job-1", persist=persist),
        )

        assert final_state is state
        assert [event.event_type for event in events] == [
            "workflow_started",
            "stage_started",
            "stage_failed",
        ]
        assert events[-1].details == {
            "error_type": "validation",
            "component": "research_source_selection",
            "reason": "Document not found.",
            "error_count": 1,
        }

    asyncio.run(run())

