"""Deterministic routing logic for the research workflow."""

import logging

from enum import Enum
from typing import Literal

from src.config import Settings
from src.graph.graph_state import WorkflowState
from src.models import (
    CoverageStatus,
    ErrorType,
    ResearchError,
    ResearchStopReason,
)


logger = logging.getLogger(__name__)


class TerminalCheckRoute(str, Enum):
    """
    Generic transition after a node that may create terminal errors.
    """

    CONTINUE = "continue"
    TERMINAL_ERROR = "terminal_error"


TerminalCheckRouteName = Literal[
    "continue",
    "terminal_error",
]


class ResearchRoute(str, Enum):
    """
    Possible transitions after research coverage evaluation.
    """

    CONTINUE_RESEARCH = (
        "continue_research"
    )

    WRITE_REPORT = (
        "write_report"
    )

    TERMINAL_ERROR = (
        "terminal_error"
    )


ResearchRouteName = Literal[
    "continue_research",
    "write_report",
    "terminal_error",
]


def has_terminal_error(
    state: WorkflowState,
) -> bool:
    """
    Returns True when workflow state contains an unrecoverable error.
    """

    return any(
        not error.recoverable
        for error in state["errors"]
    )


def route_after_terminal_check(
    state: WorkflowState,
) -> TerminalCheckRouteName:
    """
    Routes execution based solely on terminal-error presence.

    This function is pure and never mutates graph state.
    """

    if has_terminal_error(state):
        return (
            TerminalCheckRoute
            .TERMINAL_ERROR
            .value
        )

    return (
        TerminalCheckRoute
        .CONTINUE
        .value
    )


def _determine_stop_reason(
    *,
    state: WorkflowState,
    settings: Settings,
) -> ResearchStopReason | None:
    """
    Determines whether autonomous research should stop.

    Coverage sufficiency takes precedence over configured limits because
    successful coverage best describes why research completed.
    """

    coverage = (
        state["research"]["coverage"]
    )

    if coverage is None:
        return None

    if (
        coverage.status
        == CoverageStatus.SUFFICIENT
    ):
        return (
            ResearchStopReason
            .COVERAGE_SUFFICIENT
        )

    if (
        state["research"]["iteration_count"]
        >= settings
        .research_limits
        .max_iterations
    ):
        return (
            ResearchStopReason
            .MAX_ITERATIONS
        )

    total_queries = len(
        state["research"]["search_queries"]
    )

    if (
        total_queries
        >= settings
        .research_limits
        .max_total_queries_per_job
    ):
        return (
            ResearchStopReason
            .QUERY_BUDGET_EXHAUSTED
        )

    return None


def research_routing_node(
    state: WorkflowState,
    *,
    settings: Settings,
) -> dict:
    """
    Records why research should stop or indicates another pass is allowed.

    The node updates state but does not select graph edges directly.
    """

    if has_terminal_error(state):
        logger.warning(
            (
                "Research routing detected "
                "a terminal error."
            ),
            extra={
                "event": (
                    "research_terminal_error_detected"
                ),
                "job_id": state["job_id"],
                "research_session_id": state[
                    "research_session_id"
                ],
                "node": "research_routing",
                "iteration": state[
                    "research"
                ]["iteration_count"],
            },
        )

        return {}

    coverage = (
        state["research"]["coverage"]
    )

    if coverage is None:
        error = ResearchError(
            error_type=(
                ErrorType.COVERAGE
            ),
            component=(
                "research_routing"
            ),
            message=(
                "Research routing could not "
                "continue because no coverage "
                "result was available."
            ),
            recoverable=False,
        )

        logger.error(
            "Research routing found no coverage result.",
            extra={
                "event": (
                    "research_routing_missing_coverage"
                ),
                "job_id": state["job_id"],
                "research_session_id": state[
                    "research_session_id"
                ],
                "node": "research_routing",
                "iteration": state[
                    "research"
                ]["iteration_count"],
                "error_type": (
                    ErrorType
                    .COVERAGE
                    .value
                ),
            },
        )

        return {
            "errors": [
                *state["errors"],
                error,
            ]
        }

    stop_reason = (
        _determine_stop_reason(
            state=state,
            settings=settings,
        )
    )

    if stop_reason is None:
        logger.info(
            "Research will continue.",
            extra={
                "event": (
                    "research_continue"
                ),
                "job_id": state["job_id"],
                "research_session_id": state[
                    "research_session_id"
                ],
                "node": "research_routing",
                "iteration": state[
                    "research"
                ]["iteration_count"],
            },
        )

        return {
            "research": {
                **state["research"],
                "stop_reason": None,
            }
        }

    logger.info(
        "Research will stop.",
        extra={
            "event": "research_stop",
            "job_id": state["job_id"],
            "research_session_id": state[
                "research_session_id"
            ],
            "node": "research_routing",
            "iteration": state[
                "research"
            ]["iteration_count"],
        },
    )

    return {
        "research": {
            **state["research"],
            "stop_reason": stop_reason,
        }
    }


def route_after_research(
    state: WorkflowState,
) -> ResearchRouteName:
    """
    Selects the next graph edge after research routing.

    This function is pure and performs no state mutation.
    """

    if has_terminal_error(state):
        return (
            ResearchRoute
            .TERMINAL_ERROR
            .value
        )

    if (
        state["research"]["stop_reason"]
        is not None
    ):
        return (
            ResearchRoute
            .WRITE_REPORT
            .value
        )

    return (
        ResearchRoute
        .CONTINUE_RESEARCH
        .value
    )


