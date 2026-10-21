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
    "writer_prepare_evidence", "writer_synthesis", "writer_critic", "writer_citation_checker",
    "finalize_report",
})

_DETAIL_LIMIT = 100
_PUBLIC_PROVIDERS = ("tavily", "brave", "arxiv", "wikipedia")


def _provider_counts(value: Any) -> dict[str, int]:
    return {
        name: count for name in _PUBLIC_PROVIDERS
        if isinstance(value, dict) and isinstance(count := value.get(name), int) and count >= 0
    }


def _count(value: Any) -> int:
    return value if isinstance(value, int) and value >= 0 else 0


def _field(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _text(value: Any, limit: int = 1000) -> str:
    """Bound explicitly allowed display text, never serialize arbitrary objects."""
    if not isinstance(value, str):
        return ""
    return value if len(value) <= limit else value[:limit] + "…"


def _question_summary(question: Any, questions_by_id: dict) -> dict[str, Any]:
    parent_id = _field(question, "parent_subquestion_id")
    return {
        "id": _text(_field(question, "id"), 120),
        "question": _text(_field(question, "question")),
        "objective": _text(_field(question, "objective")),
        "parent_question": _text(_field(questions_by_id.get(parent_id), "question")),
    }


def _safe_details(
    node: str,
    output: Any,
    *,
    iteration: int | None = None,
    private_sources: bool = False,
) -> dict[str, Any]:
    """Whitelist bounded summaries/public previews; never expose private text."""

    if not isinstance(output, dict):
        return {}

    research = output.get("research", {})
    original_questions = _field(research, "subquestions", [])
    tasks = _field(research, "active_subquestions")
    if tasks is None:
        tasks = original_questions
    questions_by_id = {
        _field(item, "id"): item for item in [
            *original_questions, *_field(research, "planning_history", []), *tasks,
        ]
    }

    if node in {"research_planner", "research_query_generator"}:
        reason = _field(research, "stop_reason")
        details = {
            "active_subquestion_count": len(tasks),
            "stop_reason": getattr(reason, "value", reason),
            "reason": str(getattr(reason, "value", reason) or "").replace("_", " "),
            "gap_count": len(_field(output.get("research_control"), "pending_gaps", [])),
        }
        if node == "research_planner":
            details["subquestions"] = [_question_summary(item, questions_by_id) for item in tasks[:_DETAIL_LIMIT]]
            details["details_truncated"] = len(tasks) > _DETAIL_LIMIT
        else:
            queries = [
                item for item in _field(research, "search_queries", [])
                if iteration is None or _field(item, "iteration") == iteration
            ]
            grouped: dict[str, dict] = {}
            for query in queries[:_DETAIL_LIMIT]:
                task_id = _field(query, "subquestion_id")
                group = grouped.setdefault(task_id, {
                    **_question_summary(questions_by_id.get(task_id), questions_by_id),
                    "subquestion_id": _text(task_id, 120), "queries": [],
                })
                group["queries"].append({
                    "query": _text(_field(query, "query")),
                    "purpose": _text(_field(query, "purpose")),
                })
            details.update({
                "query_count": len(queries), "query_groups": list(grouped.values()),
                "details_truncated": len(queries) > _DETAIL_LIMIT,
            })
        return details

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

        details = {
            "result_count": len(results),
            "provider_counts": provider_counts,
            "failed_branches": sorted(failed_branches),
            "partial_failure_count": len(failed_branches),
            "web_fallback_activated": fallback_activated,
        }
        stats = _field(research, "search_stats", {})
        if _field(stats, "iteration") == iteration:
            discovered = _provider_counts(_field(stats, "provider_counts", {}))
            failures = [name for name in _field(stats, "failed_branches", []) if name in {"web", "arxiv", "wikipedia"}]
            details.update({
                "count_scope": "round",
                "discovered_provider_counts": discovered,
                "discovered_count": sum(discovered.values()),
                "failed_branches": failures,
                "partial_failure_count": len(failures),
                "web_fallback_activated": bool(_field(stats, "web_fallback_activated", False)),
                "failed_query_count": _field(stats, "failed_query_count", 0),
            })
            if isinstance(count := _field(stats, "executed_query_count"), int):
                details["executed_query_count"] = count
        return details

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

        details = {
            "source_count": len(sources),
            "provider_counts": provider_counts,
            "extraction_fallback_count": extraction_fallback_count,
        }
        quality = []
        for source in sources:
            if _field(source, "source_type") == "document":
                continue
            metadata = _field(source, "provider_metadata", {})
            content_source = _field(metadata, "content_source")
            if content_source not in {"pdf", "abstract"} and not _field(metadata, "content_truncated"):
                continue
            quality.append({
                "title": _text(_field(source, "title"), 250),
                "content_source": content_source if content_source in {"pdf", "abstract"} else "page",
                "total_pages": _field(metadata, "total_pages") if isinstance(_field(metadata, "total_pages"), int) else None,
                "pages_processed": _field(metadata, "pages_processed") if isinstance(_field(metadata, "pages_processed"), int) else None,
                "pdf_truncated": _field(metadata, "pdf_truncated") is True,
                "content_truncated": _field(metadata, "content_truncated") is True,
            })
        if quality:
            details["source_quality"] = quality[:_DETAIL_LIMIT]
        stats = _field(research, "extraction_stats", {})
        if _field(stats, "iteration") == iteration:
            details["extracted_provider_counts"] = _provider_counts(_field(stats, "extracted_provider_counts", {}))
            for key in (
                "extracted_count", "reused_count", "attempted_count", "deferred_count", "carried_over_count",
                "gate_passed_count", "gate_kept_min_count", "gate_rejected_count",
            ):
                value = _field(stats, key)
                if isinstance(value, int) and value >= 0:
                    details[key] = value
        return details

    if node == "research_retrieval":
        research = output.get("research", {})
        web_results = _field(research, "web_retrieval_results", [])
        document_results = _field(
            research,
            "document_retrieval_results",
            [],
        )

        def summaries(results: list) -> list[dict]:
            return [{
                "question": _text(_field(questions_by_id.get(_field(result, "subquestion_id")), "question")),
                "passage_count": len(_field(result, "chunks", [])),
                "estimated_tokens": _count(_field(result, "estimated_tokens", 0)),
                "query_matches": [{
                    "query": _text(_field(match, "query")),
                    "retrieved_count": _count(_field(match, "retrieved_count", 0)),
                    "selected_count": _count(_field(match, "selected_count", 0)),
                } for match in _field(result, "query_matches", [])[:_DETAIL_LIMIT]],
            } for result in results[:_DETAIL_LIMIT]]

        # Query-level counts explain context selection without persisting passages,
        # private filenames, chunk IDs, owner IDs, or storage paths.
        return {
            "web_groups": summaries(web_results),
            "document_groups": summaries(document_results),
            "details_truncated": any(
                len(results) > _DETAIL_LIMIT or any(len(_field(result, "query_matches", [])) > _DETAIL_LIMIT for result in results)
                for results in (web_results, document_results)
            ),
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
        assessments = _field(coverage, "assessments", [])
        gaps = _field(coverage, "gaps", [])
        counts = {name: 0 for name in ("supported", "partial", "missing", "conflicting")}
        for item in assessments:
            value = _field(item, "status")
            value = getattr(value, "value", value)
            if value in counts:
                counts[value] += 1
        return {
            "coverage_status": getattr(status, "value", status),
            "gap_count": len(gaps),
            "assessment_count": len(assessments),
            "assessment_counts": counts,
            "assessments": [{
                "question": _text(_field(questions_by_id.get(_field(item, "subquestion_id")), "question")),
                "status": getattr(_field(item, "status"), "value", _field(item, "status")),
                "evidence_strength": getattr(_field(item, "evidence_strength"), "value", _field(item, "evidence_strength")),
                "supporting_evidence_count": len(set(_field(item, "evidence_ids", []))),
                "reason": _text(_field(item, "reason")),
            } for item in assessments[:_DETAIL_LIMIT]],
            "gaps": [{
                "question": _text(_field(questions_by_id.get(_field(gap, "subquestion_id")), "question")),
                "description": _text(_field(gap, "description")),
                "priority": _field(gap, "priority"),
                "blocking": bool(_field(gap, "blocking", False)),
            } for gap in gaps[:_DETAIL_LIMIT]],
            "details_truncated": len(assessments) > _DETAIL_LIMIT or len(gaps) > _DETAIL_LIMIT,
        }

    if node == "research_evidence_extractor":
        stats = _field(research, "evidence_stats", {})
        if iteration is not None and _field(stats, "iteration") != iteration:
            return {}
        details = {
            key: _field(stats, key) for key in (
                "scored_chunk_count", "accepted_chunk_count", "low_relevance_chunk_count",
                "capacity_skipped_chunk_count", "missing_chunk_count", "unknown_chunk_count",
                "duplicate_chunk_count", "failed_batch_count", "total_evidence_count",
                "min_evidence_relevance", "capacity_reached",
            ) if isinstance(_field(stats, key), (int, float, bool))
        }
        audit = _field(research, "evidence_audit", [])
        groups = []
        remaining = _DETAIL_LIMIT
        decisions = {"accepted", "below_threshold", "capacity_skipped", "previously_accepted",
                     "not_scored", "not_scored_capacity", "scoring_failed", "missing_score", "duplicate_score"}
        for group in audit[:_DETAIL_LIMIT]:
            passages = []
            for item in _field(group, "passages", [])[:remaining]:
                private = _field(item, "source_type") != "web"
                score = _field(item, "relevance_score")
                decision = _field(item, "decision")
                passages.append({
                    "chunk_id": "Private passage" if private else _text(_field(item, "chunk_id"), 120),
                    "source_type": "document" if private else "web",
                    "title": "Private document" if private else _text(_field(item, "title"), 250),
                    "uri": "" if private else _text(_field(item, "uri")),
                    "preview": "" if private else _text(_field(item, "preview"), 800),
                    "preview_truncated": not private and (
                        bool(_field(item, "preview_truncated")) or len(str(_field(item, "preview", ""))) > 800
                    ),
                    "relevance_score": score if isinstance(score, (int, float)) and 0 <= score <= 1 else None,
                    "decision": decision if decision in decisions else "not_scored",
                    "queries": [_text(query) for query in _field(item, "queries", [])[:_DETAIL_LIMIT]],
                })
            remaining -= len(passages)
            groups.append({"question": _text(_field(group, "question")), "passages": passages})
            if not remaining:
                break
        details.update({"scoring_groups": groups, "details_truncated":
                        sum(len(_field(group, "passages", [])) for group in audit) > _DETAIL_LIMIT})
        return details

    if node in {"writer_prepare_evidence", "finalize_report"}:
        writer = output.get("writer", {})
        counts = _field(writer, "source_stats", {})
        lineage = {key: _count(_field(counts, key)) for key in (
            "collected", "retrieved", "accepted", "supplied_to_writer",
        ) if isinstance(_field(counts, key), int)}
        report = _field(writer, "final_report")
        if report is not None:
            lineage["cited"] = len(_field(report, "sources", []))
        details: dict[str, Any] = {"source_lineage": lineage} if lineage else {}
        if node == "writer_prepare_evidence":
            groups = _field(writer, "evidence_summary", []) or []
            details["evidence_count"] = len(_field(writer, "evidence_ids", []))
            details["evidence_groups"] = [{
                "question": _text(_field(group, "question"), 300),
                "evidence_count": _count(_field(group, "evidence_count")),
                "source_count": _count(_field(group, "source_count")),
            } for group in groups[:_DETAIL_LIMIT]]
        return details

    if node == "writer_synthesis":
        writer = output.get("writer", {})
        draft = _field(writer, "draft_report")
        revision = _count(_field(writer, "revision_count")) > 0
        if draft is None:
            return {"revision": revision}
        findings = _field(draft, "findings", [])
        return {
            "revision": revision,
            "title": _text(_field(draft, "title"), 300),
            "finding_count": len(findings),
            "limitation_count": len(_field(draft, "limitations", [])),
            "evidence_count": len(_field(writer, "evidence_ids", [])),
            "findings": [{
                "section": _text(_field(finding, "section"), 200),
                "title": _text(_field(finding, "title"), 300),
                "citation_count": len(_field(finding, "evidence_ids", [])),
            } for finding in findings[:_DETAIL_LIMIT]],
        }

    if node == "writer_critic":
        review = _field(output.get("writer", {}), "critic_review")
        if review is None:
            return {}
        issues = _field(review, "issues", [])
        issue_types: dict[str, int] = {}
        for issue in issues[:_DETAIL_LIMIT]:
            issue_type = str(_field(issue, "issue_type", "unknown"))
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        details = {
            "verdict": _field(review, "verdict"),
            "issue_count": len(issues),
            "issue_types": issue_types,
        }
        # Instructions may quote private evidence, so document jobs stream counts only.
        if not private_sources:
            details["issues"] = [{
                "issue_type": str(_field(issue, "issue_type", "unknown")),
                "finding_index": _field(issue, "finding_index"),
                "instruction": _text(_field(issue, "instruction"), 500),
            } for issue in issues[:_DETAIL_LIMIT]]
        return details

    if node == "writer_citation_checker":
        writer = output.get("writer", {})
        report = _field(writer, "final_report")
        result = _field(writer, "citation_result")
        if report is None:
            return {}
        issues = _field(result, "issues", []) if result is not None else []
        details = {
            "verified_finding_count": len(_field(report, "findings", [])),
            "unverified_item_count": len(_field(report, "unverified_material", [])),
            "issue_count": len(issues),
            "semantic_check_unavailable": any(
                _field(error, "component") == "writer_citation_checker"
                for error in output.get("errors", []) or []
            ),
        }
        if not private_sources:
            details["issues"] = [_text(_field(issue, "description"), 300) for issue in issues[:_DETAIL_LIMIT]]
            details["unverified_titles"] = [
                _text(_field(item, "title"), 300)
                for item in _field(report, "unverified_material", [])[:_DETAIL_LIMIT]
            ]
        return details

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
                private_sources="documents" in [
                    getattr(source, "value", source) for source in state.get("allowed_sources", [])
                ],
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

