"""Deterministic input guardrail for research workflow requests."""

import logging

from src.config import Settings
from src.graph.graph_state import WorkflowState
from src.models import (
    ErrorType,
    ResearchError,
    ResearchSource,
)


logger = logging.getLogger(__name__)


def input_guardrail_node(
    state: WorkflowState,
    *,
    settings: Settings,
) -> dict:
    """
    Validates workflow input before any research agent receives it.

    Phase 1 supports web research only. Future document and MCP source
    modes will extend this guardrail without changing downstream agents.
    """

    errors = list(
        state["errors"]
    )

    question = (
        state["question"]
        .strip()
    )

    if len(question) < 3:
        errors.append(
            ResearchError(
                error_type=(
                    ErrorType.VALIDATION
                ),
                component=(
                    "input_guardrail"
                ),
                message=(
                    "Research question must contain "
                    "at least 3 characters."
                ),
                recoverable=False,
            )
        )

    elif (len(question) > settings.research_limits.max_question_chars):
        errors.append(
            ResearchError(
                error_type=(
                    ErrorType.VALIDATION
                ),
                component=(
                    "input_guardrail"
                ),
                message=(
                    "Research question length exceeds "
                    "the configured maximum length."
                ),
                recoverable=False,
            )
        )

    allowed_sources = list(
        dict.fromkeys(
            state["allowed_sources"]
        )
    )

    if not allowed_sources:
        errors.append(
            ResearchError(
                error_type=(
                    ErrorType.VALIDATION
                ),
                component=(
                    "input_guardrail"
                ),
                message=(
                    "At least one research source "
                    "must be allowed."
                ),
                recoverable=False,
            )
        )

    unsupported_sources = [
        source
        for source in allowed_sources
        if source != ResearchSource.WEB
    ]

    if unsupported_sources:
        unsupported_names = ", ".join(
            source.value
            for source
            in unsupported_sources
        )

        errors.append(
            ResearchError(
                error_type=(
                    ErrorType.VALIDATION
                ),
                component=(
                    "input_guardrail"
                ),
                message=(
                    "The following research sources "
                    "are not enabled in Phase 1: "
                    f"{unsupported_names}."
                ),
                recoverable=False,
            )
        )

    if errors != state["errors"]:
        logger.warning(
            "Research input rejected.",
            extra={
                "event": (
                    "input_guardrail_rejected"
                ),
                "job_id": state["job_id"],
                "research_session_id": state[
                    "research_session_id"
                ],
                "node": "input_guardrail",
                "error_type": (
                    ErrorType
                    .VALIDATION
                    .value
                ),
            },
        )

        return {
            "errors": errors,
        }

    logger.info(
        "Research input accepted.",
        extra={
            "event": (
                "input_guardrail_accepted"
            ),
            "job_id": state["job_id"],
            "research_session_id": state[
                "research_session_id"
            ],
            "node": "input_guardrail",
        },
    )

    return {
        "question": question,
        "allowed_sources": (
            allowed_sources
        ),
    }


