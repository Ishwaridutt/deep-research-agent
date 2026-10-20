"""Research coverage and evidence quality evaluation node."""

import logging
from datetime import UTC, datetime

from src.graph.graph_state import WorkflowState
from src.llm import (
    LLMCallMetadata,
    LLMRole,
    LLMService,
    LLMServiceError,
)
from src.models import (
    CoverageGap,
    CoverageResult,
    CoverageStatus,
    ErrorType,
    Evidence,
    ResearchError,
    ResearchSource,
    SubQuestion,
)
from src.models.coverage_model import (
    EvidenceStrength,
    SubquestionCoverage,
    SubquestionCoverageStatus,
)
from src.prompts.coverage_evaluator_prompt import (
    build_coverage_evaluator_messages,
)


logger = logging.getLogger(__name__)


def _group_evidence_by_subquestion(
    *,
    subquestions: list[SubQuestion],
    evidence: list[Evidence],
) -> dict[str, list[Evidence]]:
    """Group accepted evidence by its planned subquestion."""

    grouped = {subquestion.id: [] for subquestion in subquestions}
    seen_ids: set[tuple[str, str]] = set()

    for item in evidence:
        key = (item.subquestion_id, item.id)
        if item.subquestion_id not in grouped or key in seen_ids:
            continue

        grouped[item.subquestion_id].append(item)
        seen_ids.add(key)

    return grouped


def _build_missing_evidence_coverage(
    subquestions: list[SubQuestion],
) -> CoverageResult:
    """Return insufficient coverage when no evidence exists."""

    return CoverageResult(
        status=CoverageStatus.INSUFFICIENT,
        gaps=[
            CoverageGap(
                subquestion_id=subquestion.id,
                description=(
                    "No accepted evidence is available "
                    "for this subquestion."
                ),
            )
            for subquestion in subquestions
        ],
        assessments=[
            SubquestionCoverage(
                subquestion_id=subquestion.id,
                status=SubquestionCoverageStatus.MISSING,
                evidence_strength=EvidenceStrength.UNKNOWN,
                evidence_ids=[],
                reason="No accepted evidence is available.",
            )
            for subquestion in subquestions
        ],
    )


def _normalize_coverage_result(
    *,
    coverage: CoverageResult,
    subquestions: list[SubQuestion],
    evidence_by_subquestion: dict[str, list[Evidence]],
) -> CoverageResult:
    """Validate references without turning every qualification into more research."""

    valid_ids = {
        subquestion.id
        for subquestion in subquestions
    }

    # Keep valid, distinct gaps.
    gaps: list[CoverageGap] = []
    seen_gaps: set[tuple[str, str]] = set()

    for gap in coverage.gaps:
        description = gap.description.strip()
        key = (gap.subquestion_id, description.casefold())

        if (
            gap.subquestion_id not in valid_ids
            or not description
            or key in seen_gaps
        ):
            continue

        seen_gaps.add(key)

        gaps.append(gap.model_copy(update={"description": description}))

    # Only accept assessments for planned subquestions.
    assessments_by_id: dict[str, SubquestionCoverage] = {}
    duplicate_assessment_ids: set[str] = set()

    for assessment in coverage.assessments:
        subquestion_id = assessment.subquestion_id

        if subquestion_id not in valid_ids:
            continue

        if subquestion_id in assessments_by_id:
            duplicate_assessment_ids.add(subquestion_id)
            continue

        assessments_by_id[subquestion_id] = assessment

    normalized_assessments: list[SubquestionCoverage] = []
    integrity_ok = bool(subquestions)

    for subquestion in subquestions:
        subquestion_id = subquestion.id

        accepted_evidence_ids = {
            item.id
            for item in evidence_by_subquestion.get(
                subquestion_id,
                [],
            )
        }

        assessment = assessments_by_id.get(subquestion_id)

        if assessment is None or subquestion_id in duplicate_assessment_ids:
            integrity_ok = False
            status = (
                SubquestionCoverageStatus.PARTIAL if accepted_evidence_ids
                else SubquestionCoverageStatus.MISSING
            )
            strength = EvidenceStrength.UNKNOWN
            evidence_ids = []
            reason = "A unique coverage assessment was not returned for this subquestion."
        elif not accepted_evidence_ids:
            # An honestly missing peripheral objective may coexist with a useful
            # report, but claiming support without evidence is invalid output.
            if assessment.status == SubquestionCoverageStatus.SUPPORTED or assessment.evidence_ids:
                integrity_ok = False
            status = SubquestionCoverageStatus.MISSING
            strength = EvidenceStrength.UNKNOWN
            evidence_ids = []
            reason = "No accepted evidence is available."
        else:
            status = assessment.status
            strength = assessment.evidence_strength
            reason = assessment.reason.strip() or "No coverage explanation was supplied."

            # Discard invented or incorrectly assigned evidence IDs.
            evidence_ids = list(
                dict.fromkeys(
                    evidence_id
                    for evidence_id in assessment.evidence_ids
                    if evidence_id in accepted_evidence_ids
                )
            )

            if set(assessment.evidence_ids) - accepted_evidence_ids or (
                status == SubquestionCoverageStatus.SUPPORTED and not evidence_ids
            ):
                integrity_ok = False
                status = SubquestionCoverageStatus.PARTIAL
                reason = "Supporting evidence references were invalid or missing. " + reason

        normalized_assessments.append(
            SubquestionCoverage(
                subquestion_id=subquestion_id,
                status=status,
                evidence_strength=strength,
                evidence_ids=evidence_ids,
                reason=reason,
            )
        )

    # Overall readiness is the model's judgment of the core request, not an
    # all-objectives checklist. Retain minor gaps for the writer as limitations.
    has_report_evidence = any(
        assessment.evidence_ids and assessment.status != SubquestionCoverageStatus.MISSING
        for assessment in normalized_assessments
    )
    is_sufficient = (
        coverage.status == CoverageStatus.SUFFICIENT
        and integrity_ok
        and has_report_evidence
    )

    return CoverageResult(
        status=(
            CoverageStatus.SUFFICIENT
            if is_sufficient
            else CoverageStatus.INSUFFICIENT
        ),
        gaps=gaps,
        assessments=normalized_assessments,
    )


def _evaluation_error(
    *,
    state: WorkflowState,
    iteration: int,
    message: str,
) -> dict:
    """Build the state update for a terminal evaluation error."""

    error = ResearchError(
        error_type=ErrorType.COVERAGE,
        component="research_coverage_evaluator",
        message=message,
        recoverable=False,
    )

    return {
        "research": {
            **state["research"],
            "iteration_count": iteration,
        },
        "errors": [
            *state["errors"],
            error,
        ],
    }


async def research_coverage_evaluator_node(
    state: WorkflowState,
    *,
    llm_service: LLMService,
) -> dict:
    """Evaluate accumulated evidence against the research plan."""

    iteration = state["research"]["iteration_count"] + 1
    subquestions = state["research"]["subquestions"]
    evidence = state["research"]["evidence"]

    log_context = {
        "job_id": state["job_id"],
        "research_session_id": state["research_session_id"],
        "node": "research_coverage_evaluator",
        "iteration": iteration,
    }
    logger.info(
        "Research coverage evaluation started.",
        extra={**log_context, "event": "coverage_evaluation_started"},
    )

    if not subquestions:
        logger.error(
            "Research coverage evaluation has no subquestions.", extra=log_context,
        )

        return _evaluation_error(
            state=state,
            iteration=iteration,
            message=(
                "Research coverage cannot be evaluated "
                "because the research plan is empty."
            ),
        )

    grouped_evidence = _group_evidence_by_subquestion(
        subquestions=subquestions,
        evidence=evidence,
    )

    # Skip the LLM only when no subquestion has any evidence.
    if not any(grouped_evidence.values()):
        coverage = _build_missing_evidence_coverage(subquestions)
    else:
        messages = build_coverage_evaluator_messages(
            original_question=state["question"],
            subquestions=subquestions,
            evidence_by_subquestion=grouped_evidence,
            current_date=datetime.now(UTC).date().isoformat(),
            previous_coverage=state["research"]["coverage"],
        )

        metadata = LLMCallMetadata(
            **log_context,
            contains_private_data=ResearchSource.DOCUMENTS in state["allowed_sources"],
        )

        print('\ncoverage eval prompt--------->')
        print(messages)

        try:
            raw_coverage = await llm_service.invoke_structured(
                role=LLMRole.COVERAGE_EVALUATOR,
                messages=messages,
                output_schema=CoverageResult,
                metadata=metadata,
            )

        except LLMServiceError as exc:
            print("Error in coverage eval: ", exc)
            logger.exception(
                "Research coverage evaluation failed.",
                extra={
                    **log_context, "event": "coverage_evaluation_failed",
                    "error_type": ErrorType.COVERAGE.value,
                },
            )

            return _evaluation_error(
                state=state,
                iteration=iteration,
                message=str(exc),
            )
        print('\n raw coverage eval response: ')
        print(raw_coverage)

        coverage = _normalize_coverage_result(
            coverage=raw_coverage,
            subquestions=subquestions,
            evidence_by_subquestion=grouped_evidence,
        )
        if raw_coverage.status == CoverageStatus.SUFFICIENT and coverage.status != raw_coverage.status:
            logger.warning(
                "Coverage readiness rejected: assessments or evidence references were invalid or lacked usable support.",
                extra={**log_context, "event": "coverage_readiness_rejected"},
            )

        print('\n coverage eval response: ')
        print(coverage)

    logger.info(
        "Research coverage evaluation completed.",
        extra={
            "event": "coverage_evaluation_completed",
            **log_context,
            "coverage_status": coverage.status.value,
            "gap_count": len(coverage.gaps),
            "assessment_count": len(coverage.assessments),
        },
    )

    return {
        "research": {
            **state["research"],
            "iteration_count": iteration,
            "coverage": coverage,
        }
    }
