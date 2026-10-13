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


def _safe_details(node: str, output: Any) -> dict[str, Any]:
    """Never leak prompts, webpage bodies, evidence, or final reports into SSE."""

    if not isinstance(output, dict):
        return {}

    if node == "research_source_selection":
        source_mode = output.get("source_mode")
        return {
            "source_mode": getattr(source_mode, "value", source_mode),
            "document_count": len(output.get("selected_document_ids", [])),
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
            details = _safe_details(stage, event.get("data", {}).get("output"))
            await recorder.emit(
                "stage_completed", stage, "completed", iteration=iteration, details=details,
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

