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
    Evidence,
    ResearchError,
    ResearchStopReason,
)
from src.prompts.writer_prompt import build_writer_messages


logger = logging.getLogger(__name__)


def _resolve_selected_evidence(
    state: WorkflowState,
) -> list[Evidence]:
    """Resolves selected Writer evidence IDs to authoritative Evidence objects."""

    evidence_by_id = {
        evidence.id: evidence
        for evidence in state["research"]["evidence"]
    }

    resolved_evidence: list[Evidence] = []
    seen_ids: set[str] = set()

    for evidence_id in state["writer"]["evidence_ids"]:
        if evidence_id in seen_ids:
            continue

        evidence = evidence_by_id.get(evidence_id)

        if evidence is None:
            continue

        seen_ids.add(evidence_id)
        resolved_evidence.append(evidence)

    return resolved_evidence


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
            f"Research stopped with reason: {stop_reason.value}."
        )

    limitations.extend(
        (
            f"Unresolved research gap for {gap.subquestion_id}: "
            f"{gap.description}"
        )
        for gap in coverage_gaps
    )

    return DraftReport(
        executive_summary=(
            "The research process did not obtain enough usable evidence "
            "to produce supported findings."
        ),
        findings=[],
        limitations=limitations,
    )


def _sanitize_draft_report(
    *,
    draft_report: DraftReport,
    allowed_evidence_ids: set[str],
) -> DraftReport:
    """Removes evidence IDs that were not supplied to the Writer."""

    sanitized_findings = []

    for finding in draft_report.findings:
        valid_evidence_ids = list(
            dict.fromkeys(
                evidence_id
                for evidence_id in finding.evidence_ids
                if evidence_id in allowed_evidence_ids
            )
        )

        sanitized_findings.append(
            finding.model_copy(
                update={
                    "evidence_ids": valid_evidence_ids,
                }
            )
        )

    return draft_report.model_copy(
        update={
            "findings": sanitized_findings,
        }
    )


async def writer_synthesis_node(
    state: WorkflowState,
    *,
    llm_service: LLMService,
) -> dict:
    """Synthesizes selected research evidence into a structured DraftReport."""

    logger.info(
        "Writer synthesis started.",
        extra={
            "event": "writer_synthesis_started",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "writer_synthesis",
            "iteration": state["research"]["iteration_count"],
        },
    )

    evidence = _resolve_selected_evidence(state)

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
                "job_id": state["job_id"],
                "research_session_id": state["research_session_id"],
                "node": "writer_synthesis",
                "iteration": state["research"]["iteration_count"],
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
        job_id=state["job_id"],
        research_session_id=state["research_session_id"],
        node="writer_synthesis",
        iteration=state["research"]["iteration_count"],
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
                "job_id": state["job_id"],
                "research_session_id": state["research_session_id"],
                "node": "writer_synthesis",
                "iteration": state["research"]["iteration_count"],
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

    # Keep unsupported evidence references visible to the citation checker,
    # but prevent references to evidence the Writer was never given.
    draft_report = _sanitize_draft_report(
        draft_report=draft_report,
        allowed_evidence_ids={
            evidence_item.id
            for evidence_item in evidence
        },
    )

    logger.info(
        "Writer synthesis completed.",
        extra={
            "event": "writer_synthesis_completed",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "writer_synthesis",
            "iteration": state["research"]["iteration_count"],
        },
    )

    return {
        "writer": {
            **state["writer"],
            "draft_report": draft_report,
        }
    }

