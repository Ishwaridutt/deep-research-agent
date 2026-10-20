"""Writer synthesis node."""

import logging

from src.graph.graph_state import WorkflowState
from src.llm import (
    LLMCallMetadata,
    LLMRole,
    LLMService,
    LLMServiceError,
)
from src.models import (
    CoverageGap,
    DraftReport,
    ErrorType,
    ResearchError,
    ResearchSource,
    ResearchStopReason,
)
from src.prompts.writer_prompt import build_writer_messages


logger = logging.getLogger(__name__)


def _build_no_evidence_report(
    *,
    stop_reason: ResearchStopReason | None,
    coverage_gaps: list[CoverageGap],
) -> DraftReport:
    """Creates a deterministic report when no usable evidence is available."""

    limitations = [
        "No usable evidence was available for grounded report synthesis."
    ]

    if stop_reason is not None:
        limitations.append(
            f"Research stopped: {stop_reason.value.replace('_', ' ')}."
        )

    limitations.extend(
        gap.description
        for gap in coverage_gaps
    )

    return DraftReport(
        title="Insufficient research evidence",
        executive_summary=(
            "The research process did not obtain enough usable evidence "
            "to produce supported findings."
        ),
        findings=[],
        limitations=limitations,
    )


async def writer_synthesis_node(
    state: WorkflowState,
    *,
    llm_service: LLMService,
) -> dict:
    """Synthesizes selected research evidence into a structured DraftReport."""

    log_context = {
        "job_id": state["job_id"],
        "research_session_id": state["research_session_id"],
        "node": "writer_synthesis",
        "iteration": state["research"]["iteration_count"],
    }
    logger.info(
        "Writer synthesis started.",
        extra={**log_context, "event": "writer_synthesis_started"},
    )
    evidence_by_id = {item.id: item for item in state["research"]["evidence"]}
    evidence = [
        evidence_by_id[identifier]
        for identifier in dict.fromkeys(state["writer"]["evidence_ids"])
        if identifier in evidence_by_id
    ]

    coverage = state["research"]["coverage"]
    coverage_gaps = list(coverage.gaps) if coverage is not None else []
    stop_reason = state["research"]["stop_reason"]

    # Avoid spending an LLM call when there is no evidence to synthesize.
    if not evidence:
        draft_report = _build_no_evidence_report(
            stop_reason=stop_reason,
            coverage_gaps=coverage_gaps,
        )

        logger.warning(
            "Writer synthesis skipped because no usable evidence was available.",
            extra={
                "event": "writer_synthesis_no_evidence",
                **log_context,
            },
        )

        return {
            "writer": {
                **state["writer"],
                "draft_report": draft_report,
            }
        }

    messages = build_writer_messages(
        original_question=state["question"],
        subquestions=state["research"]["subquestions"],
        evidence=evidence,
        coverage_gaps=coverage_gaps,
        stop_reason=stop_reason,
    )

    metadata = LLMCallMetadata(
        **log_context,
        contains_private_data=ResearchSource.DOCUMENTS in state["allowed_sources"],
    )

    try:
        draft_report = await llm_service.invoke_structured(
            role=LLMRole.WRITER,
            messages=messages,
            output_schema=DraftReport,
            metadata=metadata,
        )

    except LLMServiceError as exc:
        logger.exception(
            "Writer synthesis failed.",
            extra={
                "event": "writer_synthesis_failed",
                **log_context,
                "error_type": ErrorType.SYNTHESIS.value,
            },
        )

        error = ResearchError(
            error_type=ErrorType.SYNTHESIS,
            component="writer_synthesis",
            message=str(exc),
            recoverable=False,
        )

        return {
            "errors": [
                *state["errors"],
                error,
            ]
        }

    # The citation checker owns reference validation; retain invalid IDs so it
    # can reject unsupported blocks rather than silently repairing their references.
    logger.info(
        "Writer synthesis completed.",
        extra={
            "event": "writer_synthesis_completed",
            **log_context,
        },
    )

    return {
        "writer": {
            **state["writer"],
            "draft_report": draft_report,
        }
    }

