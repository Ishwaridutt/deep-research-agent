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

    Validate the selected web and private-document research sources.
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

    supported_sources = {
        ResearchSource.WEB,
        ResearchSource.DOCUMENTS,
    }
    unsupported_sources = [
        source
        for source in allowed_sources
        if source not in supported_sources
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
                    "are not enabled: "
                    f"{unsupported_names}."
                ),
                recoverable=False,
            )
        )

    if ResearchSource.DOCUMENTS in allowed_sources:
        if not state["owner_id"]:
            errors.append(
                ResearchError(
                    error_type=ErrorType.VALIDATION,
                    component="input_guardrail",
                    message=(
                        "X-Owner-Id is required when private documents "
                        "are selected."
                    ),
                    recoverable=False,
                )
            )

        if not state["selected_document_ids"]:
            errors.append(
                ResearchError(
                    error_type=ErrorType.VALIDATION,
                    component="input_guardrail",
                    message=(
                        "At least one private document must be selected "
                        "for document research."
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


