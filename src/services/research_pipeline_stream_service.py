"""Observe one LangGraph run and persist sanitized pipeline lifecycle events."""

import json
from collections.abc import Awaitable, Callable
from typing import Any, cast

from langgraph.graph.state import CompiledStateGraph

from src.graph.graph_state import WorkflowState
from src.models.pipeline_event import EventType, PipelineEvent


PIPELINE_NODES = frozenset({
    "input_guardrail", "research_planner", "research_query_generator",
    "research_source_selection",
    "research_web_search", "research_source_processor", "research_retrieval",
    "research_evidence_extractor", "research_coverage_evaluator", "research_routing",
    "writer_prepare_evidence", "writer_synthesis", "writer_citation_checker",
    "finalize_report",
})


def _field(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _safe_details(
    node: str,
    output: Any,
    *,
    iteration: int | None = None,
) -> dict[str, Any]:
    """Never leak prompts, webpage bodies, evidence, or final reports into SSE."""

    if not isinstance(output, dict):
        return {}

    if node in {"research_planner", "research_query_generator"}:
        research = output.get("research", {})
        reason = _field(research, "stop_reason")
        return {
            "active_subquestion_count": len(_field(research, "active_subquestions", [])),
            "stop_reason": getattr(reason, "value", reason),
            "reason": str(getattr(reason, "value", reason) or "").replace("_", " "),
            "gap_count": len(_field(output.get("research_control"), "pending_gaps", [])),
        }

    if node == "research_source_selection":
        source_mode = output.get("source_mode")
        if source_mode is None:
            return {}
        return {
            "source_mode": getattr(source_mode, "value", source_mode),
            "document_count": len(output.get("selected_document_ids", [])),
        }

    if node == "research_web_search":
        research = output.get("research", {})
        queries = _field(research, "search_queries", [])
        query_ids = {
            _field(query, "id")
            for query in queries
            if iteration is None or _field(query, "iteration") == iteration
        }
        results = [
            result for result in _field(research, "search_results", [])
            if _field(result, "query_id") in query_ids
        ]
        provider_counts: dict[str, int] = {}
        failed_branches: set[str] = set()
        fallback_activated = False
        for result in results:
            provider = _field(result, "provider")
            provider = getattr(provider, "value", provider)
            if isinstance(provider, str):
                provider_counts[provider] = provider_counts.get(provider, 0) + 1

            metadata = _field(result, "provider_metadata", {})
            diagnostics = _field(metadata, "search_diagnostics", {})
            failed_branches.update(
                branch for branch in _field(diagnostics, "failed_branches", [])
                if isinstance(branch, str)
            )
            fallback_activated = fallback_activated or bool(
                _field(diagnostics, "web_fallback_activated", False)
            )

        return {
            "result_count": len(results),
            "provider_counts": provider_counts,
            "failed_branches": sorted(failed_branches),
            "partial_failure_count": len(failed_branches),
            "web_fallback_activated": fallback_activated,
        }

    if node == "research_source_processor":
        research = output.get("research", {})
        queries = _field(research, "search_queries", [])
        query_ids = {
            _field(query, "id")
            for query in queries
            if iteration is None or _field(query, "iteration") == iteration
        }
        result_ids = {
            _field(result, "id")
            for result in _field(research, "search_results", [])
            if _field(result, "query_id") in query_ids
        }
        sources = [
            source for source in _field(research, "sources", [])
            if result_ids.intersection(_field(source, "search_result_ids", []))
        ]
        provider_counts: dict[str, int] = {}
        extraction_fallback_count = 0
        for source in sources:
            provider = _field(source, "provider")
            provider = getattr(provider, "value", provider)
            if isinstance(provider, str):
                provider_counts[provider] = provider_counts.get(provider, 0) + 1
            metadata = _field(source, "provider_metadata", {})
            if _field(metadata, "extraction_provider") == "firecrawl":
                extraction_fallback_count += 1

        return {
            "source_count": len(sources),
            "provider_counts": provider_counts,
            "extraction_fallback_count": extraction_fallback_count,
        }

    if node == "research_retrieval":
        research = output.get("research", {})
        web_results = _field(research, "web_retrieval_results", [])
        document_results = _field(
            research,
            "document_retrieval_results",
            [],
        )

        # Persist counts only; retrieved text and private provenance stay in memory.
        return {
            "web_result_count": len(web_results),
            "document_result_count": len(document_results),
            "web_chunk_count": sum(
                len(_field(result, "chunks", [])) for result in web_results
            ),
            "document_chunk_count": sum(
                len(_field(result, "chunks", []))
                for result in document_results
            ),
        }

    if node == "research_coverage_evaluator":
        coverage = _field(output.get("research", {}), "coverage")
        if coverage is None:
            return {}
        status = _field(coverage, "status")
        return {
            "coverage_status": getattr(status, "value", status),
            "gap_count": len(_field(coverage, "gaps", [])),
            "assessment_count": len(_field(coverage, "assessments", [])),
        }

    if node == "research_routing":
        control = output.get("research_control")
        decision = _field(control, "last_decision")
        if decision is None:
            return {}
        return {
            "action": _field(decision, "action"),
            "reason": str(_field(decision, "reason", ""))[:300],
            "gap_count": len(_field(control, "pending_gaps", [])),
            "budget_exhausted": bool(_field(decision, "budget_exhausted", False)),
        }

    return {}


def _safe_terminal_error_details(output: Any) -> dict[str, Any]:
    """Expose only public ResearchError fields for a failed stage."""

    if not isinstance(output, dict):
        return {}
    terminal_errors = [
        error for error in output.get("errors", [])
        if _field(error, "recoverable", True) is False
    ]
    if not terminal_errors:
        return {}

    error = terminal_errors[0]
    error_type = _field(error, "error_type")
    return {
        "error_type": getattr(error_type, "value", error_type),
        "component": str(_field(error, "component", "workflow"))[:100],
        "reason": str(_field(error, "message", "Workflow stage failed."))[:300],
        "error_count": len(terminal_errors),
    }


class PipelineRecorder:
    """Assign job-local event sequence numbers and persist each event."""

    def __init__(
        self,
        *,
        job_id: str,
        persist: Callable[[PipelineEvent], Awaitable[None]],
    ) -> None:
        self.job_id = job_id
        self.sequence = 0
        self._persist = persist

    def make(
        self,
        event_type: EventType,
        stage: str,
        status: str,
        *,
        iteration: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> PipelineEvent:
        """Create an event; terminal events use this for atomic result writes."""

        self.sequence += 1
        return PipelineEvent(
            job_id=self.job_id,
            sequence=self.sequence,
            event_type=event_type,
            stage=stage,
            status=status,
            iteration=iteration,
            details=details or {},
        )

    async def emit(
        self,
        event_type: EventType,
        stage: str,
        status: str,
        *,
        iteration: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        await self._persist(self.make(
            event_type, stage, status, iteration=iteration, details=details,
        ))


async def run_graph_with_events(
    *,
    graph: CompiledStateGraph,
    state: WorkflowState,
    config: dict[str, Any],
    recorder: PipelineRecorder,
) -> WorkflowState:
    """Run LangGraph exactly once and capture its root final output.

    v2 event ``parent_ids`` identifies the root graph; named node events
    drive the UI. Only compact, explicitly selected metadata is persisted.
    """

    await recorder.emit("workflow_started", "workflow", "running")
    root_run_id: str | None = None
    final_state: WorkflowState | None = None
    active: dict[str, tuple[str, int | None]] = {}
    current_round = state["research"]["iteration_count"] + 1

    async for event in graph.astream_events(state, config=config, version="v2"):
        name = event.get("name")
        kind = event.get("event")
        run_id = event.get("run_id")
        parent_ids = event.get("parent_ids", [])

        if kind == "on_chain_start" and not parent_ids and root_run_id is None:
            root_run_id = run_id

        if run_id == root_run_id and kind == "on_chain_end":
            output = event.get("data", {}).get("output")
            if isinstance(output, dict):
                final_state = cast(WorkflowState, output)
            continue

        # Ignore nested LLM/tool events and other non-workflow runnables.
        if name not in PIPELINE_NODES:
            continue

        if kind == "on_chain_start":
            iteration = current_round if name.startswith("research_") else None
            active[run_id] = (name, iteration)
            await recorder.emit("stage_started", name, "running", iteration=iteration)

        elif kind == "on_chain_end" and run_id in active:
            stage, iteration = active.pop(run_id)
            output = event.get("data", {}).get("output")
            details = _safe_details(
                stage,
                output,
                iteration=iteration,
            )
            failure_details = _safe_terminal_error_details(output)
            if failure_details:
                await recorder.emit(
                    "stage_failed",
                    stage,
                    "failed",
                    iteration=iteration,
                    details={**details, **failure_details},
                )
                continue

            await recorder.emit(
                "stage_completed", stage, "completed", iteration=iteration, details=details,
            )

            if stage in {"research_planner", "research_query_generator"} and details.get("stop_reason"):
                await recorder.emit(
                    "research_stopped", stage, "completed", iteration=iteration, details=details,
                )

            if stage == "research_routing":
                action = details.get("action")
                if action == "continue_research":
                    await recorder.emit(
                        "research_retry", stage, "retrying", iteration=iteration,
                        details={**details, "next_iteration": current_round + 1},
                    )
                    current_round += 1
                elif action == "write_report":
                    await recorder.emit(
                        "research_stopped", stage, "completed",
                        iteration=iteration, details=details,
                    )

        elif kind == "on_chain_error" and run_id in active:
            stage, iteration = active.pop(run_id)
            await recorder.emit("stage_failed", stage, "failed", iteration=iteration)

    if final_state is None:
        raise RuntimeError("LangGraph finished without a root final-state event.")
    return final_state


def encode_sse(event: PipelineEvent) -> str:
    """Encode an event for browser EventSource with a replayable ID."""

    payload = json.dumps(event.model_dump(mode="json"), separators=(",", ":"))
    return f"id: {event.sequence}\nevent: {event.event_type}\ndata: {payload}\n\n"

