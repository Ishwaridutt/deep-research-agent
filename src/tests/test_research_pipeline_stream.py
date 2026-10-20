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
        "web_groups": [{"question": "", "passage_count": 1, "estimated_tokens": 0, "query_matches": []}],
        "document_groups": [{"question": "", "passage_count": 2, "estimated_tokens": 0, "query_matches": []}],
        "details_truncated": False,
        "web_result_count": 1,
        "document_result_count": 1,
        "web_chunk_count": 1,
        "document_chunk_count": 2,
    }
    assert "SECRET" not in str(retrieval)
    assert "private.pdf" not in str(retrieval)


def test_retrieval_details_keep_query_lineage_without_private_passages():
    from src.models.retrieval_model import RetrievalResult
    result = RetrievalResult("task", "PRIVATE RAW CONTEXT", [], "SECRET", 50, query_matches=[{
        "query": "published research", "retrieved_count": 8, "selected_count": 3,
        "chunks": [{"content": "SECRET", "filename": "private.pdf"}],
    }])
    details = _safe_details("research_retrieval", {"research": {
        "subquestions": [{"id": "task", "question": "What works?"}],
        "document_retrieval_results": [result],
    }}, iteration=2)
    assert details["document_groups"] == [{
        "question": "What works?", "passage_count": 0, "estimated_tokens": 50,
        "query_matches": [{"query": "published research", "retrieved_count": 8, "selected_count": 3}],
    }]
    assert "SECRET" not in str(details) and "private.pdf" not in str(details)


def test_discovery_and_extraction_counts_are_current_round_and_allowlisted():
    research = {
        "search_stats": {"iteration": 2, "provider_counts": {"tavily": 10, "arxiv": 6, "wikipedia": 0, "SECRET": 8},
                         "failed_branches": ["wikipedia", "SECRET"], "failed_query_count": 1},
        "extraction_stats": {"iteration": 2, "extracted_count": 4, "attempted_count": 6,
                             "reused_count": 1, "deferred_count": 3,
                             "extracted_provider_counts": {"arxiv": 1, "tavily": 3, "SECRET": 5}},
    }
    details = _safe_details("research_web_search", {"research": research}, iteration=2)
    assert details["discovered_provider_counts"] == {"tavily": 10, "arxiv": 6, "wikipedia": 0}
    assert details["discovered_count"] == 16 and details["result_count"] == 0
    assert details["failed_branches"] == ["wikipedia"]
    assert "SECRET" not in str(details)
    assert "discovered_count" not in _safe_details("research_web_search", {"research": research}, iteration=3)
    extracted = _safe_details("research_source_processor", {"research": research}, iteration=2)
    assert extracted["extracted_count"] == 4 and extracted["deferred_count"] == 3
    assert extracted["extracted_provider_counts"] == {"tavily": 3, "arxiv": 1}
    assert "SECRET" not in str(extracted)


def test_log_and_event_search_counts_match_for_the_same_job_and_round(caplog):
    import logging
    from types import SimpleNamespace
    from src.agents.research.research_web_search_node import research_web_search_node
    from src.graph.graph_state import create_initial_workflow_state
    from src.models import ResearchSource, SearchQuery
    from src.tools.search.parallel_search import SearchResults

    class Provider:
        def __init__(self, wiki_counts):
            self.wiki_counts = iter(wiki_counts)

        async def search(self, **kwargs):
            return SearchResults([], {"provider_counts": {"tavily": 5, "arxiv": 3, "wikipedia": next(self.wiki_counts)}})

    settings = SimpleNamespace(research_limits=SimpleNamespace(max_total_queries_per_job=30, max_search_results_per_query=5),
                               search=SimpleNamespace(min_provider_score=0.7))
    caplog.set_level(logging.INFO)
    for job, round_number, wiki_counts, expected in [
        ("new-job", 1, [1] * 12, {"tavily": 60, "arxiv": 36, "wikipedia": 12}),
        ("old-job", 2, [1, 1] + [0] * 12, {"tavily": 70, "arxiv": 42, "wikipedia": 2}),
    ]:
        state = create_initial_workflow_state(job_id=job, research_session_id="session", question="Topic?", allowed_sources=[ResearchSource.WEB])
        state["research"]["iteration_count"] = round_number - 1
        state["research"]["search_queries"] = [SearchQuery(
            id=f"query-{n}", subquestion_id="task", query="research", purpose="Research", iteration=round_number,
        ) for n in range(len(wiki_counts))]
        update = asyncio.run(research_web_search_node(state, search_provider=Provider(wiki_counts), settings=settings))
        details = _safe_details("research_web_search", update, iteration=round_number)
        record = next(record for record in reversed(caplog.records) if getattr(record, "event", "") == "web_search_completed")
        assert record.job_id == job and record.iteration == round_number
        assert details["discovered_provider_counts"] == expected
        assert details["executed_query_count"] == len(wiki_counts)
        for key in ("count_scope", "executed_query_count", "discovered_provider_counts", "discovered_count", "provider_counts", "result_count"):
            assert details[key] == record.data[key]


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


def test_planner_and_queries_expose_bounded_current_tasks_not_history_or_prompts():
    research = {
        "subquestions": [{"id": "root", "question": "Original objective?", "objective": "Original scope"}],
        "active_subquestions": [{
            "id": "followup", "question": "Which outcome is missing?", "objective": "Measure the outcome",
            "parent_subquestion_id": "root", "private_text": "SECRET",
        }],
        "planning_history": [{"id": "old", "question": "HISTORICAL PRIVATE QUESTION"}],
        "search_queries": [
            {"id": "old-query", "subquestion_id": "old", "query": "OLD PRIVATE QUERY", "iteration": 1},
            {"id": "q1", "subquestion_id": "followup", "query": "current trial result", "purpose": "Find measurement", "iteration": 2},
            {"id": "q2", "subquestion_id": "followup", "query": "current outcome uncertainty", "purpose": "Find uncertainty", "iteration": 2},
        ],
        "evidence": [{"content": "SECRET EVIDENCE"}],
        "sources": [{"content": "SECRET SOURCE"}],
    }
    planner = _safe_details("research_planner", {"research": research}, iteration=2)
    queries = _safe_details("research_query_generator", {"research": research}, iteration=2)

    assert planner["subquestions"] == [{
        "id": "followup", "question": "Which outcome is missing?", "objective": "Measure the outcome",
        "parent_question": "Original objective?",
    }]
    assert queries["query_count"] == 2
    assert len(queries["query_groups"]) == 1
    assert queries["query_groups"][0]["subquestion_id"] == "followup"
    assert [item["query"] for item in queries["query_groups"][0]["queries"]] == [
        "current trial result", "current outcome uncertainty",
    ]
    assert "PRIVATE" not in str(planner) + str(queries)
    assert "SECRET" not in str(planner) + str(queries)


def test_stage_detail_limits_preserve_totals_and_signal_truncation():
    research = {
        "subquestions": [{"id": f"task-{index}", "question": "Q" * 2000, "objective": "Scope"} for index in range(105)],
        "search_queries": [{"subquestion_id": "task-0", "query": f"query-{index}", "iteration": 1} for index in range(105)],
    }
    planner = _safe_details("research_planner", {"research": research}, iteration=1)
    queries = _safe_details("research_query_generator", {"research": research}, iteration=1)
    assert planner["active_subquestion_count"] == 105
    assert len(planner["subquestions"]) == 100
    assert planner["subquestions"][0]["question"] == "Q" * 1000 + "…"
    assert planner["details_truncated"]
    assert queries["query_count"] == 105
    assert len(queries["query_groups"][0]["queries"]) == 100
    assert queries["details_truncated"]


def test_coverage_details_include_per_objective_reasons_and_limits_not_evidence_text():
    from src.models import CoverageGap, CoverageResult, SubQuestion
    from src.models.coverage_model import SubquestionCoverage

    output = {"research": {
        "subquestions": [
            SubQuestion(id="root", question="What changed?", objective="Find change"),
            SubQuestion(id="other", question="Why did it change?", objective="Explain change"),
        ],
        "evidence": [{"id": "EV-PRIVATE", "content": "SECRET"}],
        "coverage": CoverageResult(status="sufficient", assessments=[
            SubquestionCoverage(subquestion_id="root", status="supported", evidence_ids=["EV-PRIVATE"], reason="A specific measurement answers this."),
            SubquestionCoverage(subquestion_id="other", status="partial", evidence_ids=[], reason="The minor historical cause is uncertain."),
        ], gaps=[CoverageGap(subquestion_id="other", description="Historical context not provided.", priority=2)]),
    }}
    details = _safe_details("research_coverage_evaluator", output, iteration=1)
    assert details["coverage_status"] == "sufficient"
    assert details["assessment_counts"] == {"supported": 1, "partial": 1, "missing": 0, "conflicting": 0}
    assert details["assessments"][0]["question"] == "What changed?"
    assert details["assessments"][0]["supporting_evidence_count"] == 1
    assert details["assessments"][1]["reason"] == "The minor historical cause is uncertain."
    assert details["gaps"] == [{"question": "Why did it change?", "description": "Historical context not provided.", "priority": 2}]
    assert "EV-PRIVATE" not in str(details)
    assert "SECRET" not in str(details)


def test_evidence_progress_is_numeric_allowlisted_and_never_replays_stale_counts():
    stats = {
        "iteration": 2, "scored_chunk_count": 8, "accepted_chunk_count": 5,
        "low_relevance_chunk_count": 2, "capacity_skipped_chunk_count": 1,
        "total_evidence_count": 12, "min_evidence_relevance": 0.6,
        "content": "SECRET PASSAGE", "filename": "private.pdf", "prompt": "SECRET",
    }
    output = {"research": {"evidence_stats": stats, "evidence": [{"content": "SECRET"}]}}
    details = _safe_details("research_evidence_extractor", output, iteration=2)
    assert details["accepted_chunk_count"] == 5
    assert details["low_relevance_chunk_count"] == 2
    assert details["capacity_skipped_chunk_count"] == 1
    assert details["total_evidence_count"] == 12
    assert "SECRET" not in str(details)
    assert "private.pdf" not in str(details)
    assert _safe_details("research_evidence_extractor", output, iteration=3) == {}


def test_scoring_previews_are_bounded_allowlisted_and_private_text_is_redacted_again():
    public = {"chunk_id": "public:1", "title": "Study", "source_type": "web",
              "preview": "a" * 5000, "relevance_score": .9, "decision": "accepted", "owner_id": "SECRET"}
    private = {**public, "chunk_id": "SECRET", "source_type": "document", "title": "SECRET.pdf",
               "preview": "SECRET", "uri": "document://SECRET"}
    output = {"research": {"evidence_stats": {"iteration": 2}, "evidence_audit": [
        {"question": "Outcomes?", "passages": [public, private] * 60},
    ]}}
    details = _safe_details("research_evidence_extractor", output, iteration=2)
    rows = details["scoring_groups"][0]["passages"]
    assert len(rows) == 100 and details["details_truncated"]
    assert len(rows[0]["preview"]) == 801
    assert rows[1]["title"] == "Private document" and rows[1]["preview"] == ""
    assert "SECRET" not in str(details)
    assert _safe_details("research_evidence_extractor", output, iteration=3) == {}


def test_writer_source_lineage_reports_counts_without_source_bodies():
    output = {"writer": {"source_stats": {"collected": 40, "retrieved": 17, "accepted": 11,
                                          "supplied_to_writer": 9, "SECRET": 1},
                         "final_report": {"sources": [{"content": "SECRET"}] * 8}}}
    details = _safe_details("finalize_report", output)
    assert details == {"source_lineage": {"collected": 40, "retrieved": 17, "accepted": 11,
                                         "supplied_to_writer": 9, "cited": 8}}


def test_completed_research_details_survive_persistence_and_sse_replay(tmp_path):
    from src.persistence import ResearchJobStore
    from src.models.pipeline_event import PipelineEvent

    async def run():
        store = ResearchJobStore(tmp_path / "details.sqlite3")
        await store.initialize()
        try:
            await store.create_job(job_id="job-1", research_session_id="session-1", question="Test", allowed_sources=["web"])
            details = _safe_details("research_planner", {"research": {
                "active_subquestions": [{"id": "task", "question": "What changed?", "objective": "Find measurements"}],
            }}, iteration=1)
            event = PipelineEvent(job_id="job-1", sequence=1, event_type="stage_completed",
                                  stage="research_planner", status="completed", iteration=1, details=details)
            await store.append_event(event)
            replay = await store.get_events("job-1", after=0)
            assert replay[0].details == details
            assert "What changed?" in encode_sse(replay[0])
            assert await store.get_events("job-1", after=1) == []
        finally:
            await store.close()

    asyncio.run(run())


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


def test_planner_early_stop_emits_safe_research_stopped_event():
    class EarlyStopGraph(FakeGraph):
        async def astream_events(self, initial_state, *, config, version):
            async for event in super().astream_events(initial_state, config=config, version=version):
                if event["name"] == "research_routing":
                    continue
                if event["name"] == "research_planner" and event["event"] == "on_chain_end":
                    event["data"]["output"] = {
                        "research": {
                            "active_subquestions": [], "stop_reason": "no_actionable_gaps",
                            "planning_history": [{"question": "PRIVATE QUESTION"}],
                        },
                        "research_control": {"pending_gaps": ["PRIVATE GAP"]},
                    }
                yield event

    async def run():
        state = {"job_id": "job-1", "research": {"iteration_count": 1}}
        events = []

        async def persist(event):
            events.append(event)

        final = await run_graph_with_events(
            graph=EarlyStopGraph(state), state=state, config={"recursion_limit": 50},
            recorder=PipelineRecorder(job_id="job-1", persist=persist),
        )
        assert final is state
        stop = events[-1]
        assert stop.event_type == "research_stopped"
        assert stop.iteration == 2
        assert stop.details["stop_reason"] == "no_actionable_gaps"
        assert stop.details["gap_count"] == 1
        assert "PRIVATE" not in "".join(encode_sse(event) for event in events)

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

