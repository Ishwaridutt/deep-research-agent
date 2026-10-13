"""Deterministic routing and execution control for research."""

import logging
from datetime import datetime, timezone
from typing import Literal

from src.config import Settings
from src.graph.graph_state import WorkflowState
from src.models import (
    CoverageResult,
    CoverageStatus,
    ErrorType,
    ResearchError,
    ResearchSource,
    ResearchStopReason,
)
from src.models.research_quality import (
    ResearchControlState,
    ResearchDecision,
    ResearchRoundRecord,
)

logger = logging.getLogger(__name__)

TerminalCheckRouteName = Literal[
    "continue",
    "terminal_error",
]

ResearchRouteName = Literal[
    "continue_research",
    "write_report",
    "terminal_error",
]

SourceSelectionRouteName = Literal[
    "web_research",
    "document_research",
    "terminal_error",
]


def has_terminal_error(state: WorkflowState) -> bool:
    """Check whether the workflow has an unrecoverable error."""

    return any(
        not error.recoverable
        for error in state["errors"]
    )


def route_after_terminal_check(
    state: WorkflowState,
) -> TerminalCheckRouteName:
    """Route based on terminal-error presence."""

    return (
        "terminal_error"
        if has_terminal_error(state)
        else "continue"
    )


def route_after_source_selection(
    state: WorkflowState,
) -> SourceSelectionRouteName:
    """Route the iteration through only the selected source pipeline."""

    if has_terminal_error(state):
        return "terminal_error"
    if ResearchSource.WEB in state["allowed_sources"]:
        return "web_research"
    return "document_research"


def _get_new_evidence_ids(
    *,
    state: WorkflowState,
    coverage: CoverageResult,
    previous_rounds: list[ResearchRoundRecord],
) -> list[str]:
    """
    Identify newly referenced evidence.

    Only count evidence IDs that:
    - Exist in accumulated research evidence.
    - Were referenced by a validated coverage assessment.
    - Were not counted in previous rounds.
    """

    available_ids = {
        item.id
        for item in state["research"]["evidence"]
    }

    referenced_ids = {
        evidence_id
        for assessment in coverage.assessments
        for evidence_id in assessment.evidence_ids
    }

    previous_ids = {
        evidence_id
        for record in previous_rounds
        for evidence_id in record.new_evidence_ids
    }

    return sorted(
        (referenced_ids & available_ids) - previous_ids
    )


def _get_elapsed_seconds(
    control: ResearchControlState,
) -> float:
    """Calculate elapsed research execution time."""

    started_at = control.usage.started_at

    elapsed = (
        datetime.now(timezone.utc) - started_at
    ).total_seconds()

    return max(0.0, elapsed)


def _determine_stop_reason(
    *,
    state: WorkflowState,
    settings: Settings,
    control: ResearchControlState,
    consecutive_no_gain_rounds: int,
    elapsed_seconds: float,
) -> ResearchStopReason | None:
    """
    Determine whether research must stop.

    Coverage sufficiency takes precedence over budget limits.
    Configured limits are combined conservatively.
    """

    research = state["research"]
    coverage = research["coverage"]
    budget = control.budget
    usage = control.usage
    limits = settings.research_limits

    if coverage is None:
        return None

    if coverage.status == CoverageStatus.SUFFICIENT:
        return ResearchStopReason.COVERAGE_SUFFICIENT

    max_rounds = min(
        budget.max_rounds,
        limits.max_iterations,
    )

    if research["iteration_count"] >= max_rounds:
        return ResearchStopReason.MAX_ITERATIONS

    max_queries = min(
        budget.max_search_queries,
        limits.max_total_queries_per_job,
    )

    # Generated queries provide a conservative fallback until
    # the search node records actual provider executions.
    queries_used = max(
        usage.queries_executed,
        len(research["search_queries"]),
    )

    if queries_used >= max_queries:
        return ResearchStopReason.QUERY_BUDGET_EXHAUSTED

    if (
        usage.page_extraction_attempts
        >= budget.max_extracted_pages
    ):
        return ResearchStopReason.EXTRACTION_BUDGET_EXHAUSTED

    if elapsed_seconds >= budget.max_research_seconds:
        return ResearchStopReason.TIME_BUDGET_EXHAUSTED

    # Allow the initial research round to establish a baseline.
    if (
        research["iteration_count"] > 1
        and consecutive_no_gain_rounds >= budget.max_consecutive_no_gain_rounds
    ):
        return ResearchStopReason.NO_EVIDENCE_IMPROVEMENT

    if not coverage.gaps:
        return ResearchStopReason.NO_ACTIONABLE_GAPS

    return None


def _build_research_decision(
    *,
    coverage: CoverageResult,
    stop_reason: ResearchStopReason | None,
) -> ResearchDecision:
    """Build an explicit routing decision."""

    if stop_reason is None:
        return ResearchDecision(
            action="continue_research",
            reason=(
                "Unresolved research gaps remain and "
                "execution budgets permit another iteration."
            ),
            has_unresolved_gaps=True,
            budget_exhausted=False,
        )

    budget_stop_reasons = {
        ResearchStopReason.MAX_ITERATIONS,
        ResearchStopReason.QUERY_BUDGET_EXHAUSTED,
        ResearchStopReason.EXTRACTION_BUDGET_EXHAUSTED,
        ResearchStopReason.TIME_BUDGET_EXHAUSTED,
        ResearchStopReason.NO_EVIDENCE_IMPROVEMENT,
    }

    return ResearchDecision(
        action="write_report",
        reason=stop_reason.value.replace("_", " "),
        has_unresolved_gaps=(
            coverage.status == CoverageStatus.INSUFFICIENT
        ),
        budget_exhausted=(
            stop_reason in budget_stop_reasons
        ),
    )


def research_routing_node(
    state: WorkflowState,
    *,
    settings: Settings,
) -> dict:
    """
    Evaluate execution limits and record the research decision.

    Does not directly select graph edges.
    """

    research = state["research"]
    control = state["research_control"]
    iteration = research["iteration_count"]

    if has_terminal_error(state):
        decision = ResearchDecision(
            action="terminal_error",
            reason="A terminal workflow error occurred.",
        )

        return {
            "research_control": control.model_copy(
                update={"last_decision": decision}
            )
        }

    coverage = research["coverage"]

    if coverage is None:
        error = ResearchError(
            error_type=ErrorType.COVERAGE,
            component="research_routing",
            message=(
                "Research routing requires a completed "
                "coverage evaluation."
            ),
            recoverable=False,
        )

        decision = ResearchDecision(
            action="terminal_error",
            reason="Coverage evaluation is unavailable.",
        )

        return {
            "errors": [
                *state["errors"],
                error,
            ],
            "research_control": control.model_copy(
                update={"last_decision": decision}
            ),
        }

    # Ignore any existing history record for this iteration.
    # This keeps round recording idempotent.
    previous_rounds = [
        record
        for record in control.history
        if record.iteration < iteration
    ]

    new_evidence_ids = _get_new_evidence_ids(
        state=state,
        coverage=coverage,
        previous_rounds=previous_rounds,
    )

    # Do not penalize the first research round for having
    # no evidence. Count stagnation from follow-up rounds.
    if iteration <= 1 or new_evidence_ids:
        consecutive_no_gain_rounds = 0
    else:
        consecutive_no_gain_rounds = (
            control.usage.consecutive_no_gain_rounds + 1
        )

    elapsed_seconds = _get_elapsed_seconds(control)

    stop_reason = _determine_stop_reason(
        state=state,
        settings=settings,
        control=control,
        consecutive_no_gain_rounds=consecutive_no_gain_rounds,
        elapsed_seconds=elapsed_seconds,
    )

    decision = _build_research_decision(
        coverage=coverage,
        stop_reason=stop_reason,
    )

    round_record = ResearchRoundRecord(
        iteration=iteration,
        new_evidence_ids=new_evidence_ids,
        unresolved_gaps=list(coverage.gaps),
        decision=decision,
    )

    updated_usage = control.usage.model_copy(
        update={
            "consecutive_no_gain_rounds": (
                consecutive_no_gain_rounds
            ),
        }
    )

    updated_control = control.model_copy(
        update={
            "usage": updated_usage,
            "pending_gaps": list(coverage.gaps),
            "history": [
                *previous_rounds,
                round_record,
            ],
            "last_decision": decision,
        }
    )

    logger.info(
        "Research routing decision completed.",
        extra={
            "event": (
                "research_continue"
                if stop_reason is None
                else "research_stop"
            ),
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "research_routing",
            "iteration": iteration,
            "action": decision.action,
            "stop_reason": (
                stop_reason.value
                if stop_reason is not None
                else None
            ),
            "gap_count": len(coverage.gaps),
            "new_evidence_count": len(new_evidence_ids),
            "elapsed_seconds": round(elapsed_seconds, 2),
        },
    )

    return {
        "research": {
            **research,
            "stop_reason": stop_reason,
        },
        "research_control": updated_control,
    }


def route_after_research(
    state: WorkflowState,
) -> ResearchRouteName:
    """Select the next graph edge after research routing."""

    if has_terminal_error(state):
        return "terminal_error"

    decision = state["research_control"].last_decision

    if decision is None:
        return "terminal_error"

    return decision.action
