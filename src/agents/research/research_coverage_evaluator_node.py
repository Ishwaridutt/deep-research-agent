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

    grouped = {
        subquestion.id: []
        for subquestion in subquestions
    }

    seen_ids = {
        subquestion.id: set()
        for subquestion in subquestions
    }

    for item in evidence:
        subquestion_id = item.subquestion_id

        if subquestion_id not in grouped:
            continue

        if item.id in seen_ids[subquestion_id]:
            continue

        grouped[subquestion_id].append(item)
        seen_ids[subquestion_id].add(item.id)

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
    """
    Validate LLM assessments against accepted research evidence.

    A subquestion can only be supported when:
    - Accepted evidence exists.
    - The LLM supplies an assessment.
    - Referenced evidence IDs are valid for that subquestion.
    - Evidence strength is strong or moderate.
    - No unresolved gap contradicts the assessment.
    """

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

        gaps.append(
            CoverageGap(
                subquestion_id=gap.subquestion_id,
                description=description,
                priority=gap.priority,
                suggested_queries=gap.suggested_queries,
            )
        )

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

        if not accepted_evidence_ids:
            status = SubquestionCoverageStatus.MISSING
            strength = EvidenceStrength.UNKNOWN
            evidence_ids = []
            reason = "No accepted evidence is available."

        elif assessment is None:
            status = SubquestionCoverageStatus.PARTIAL
            strength = EvidenceStrength.UNKNOWN
            evidence_ids = []
            reason = (
                "No valid quality assessment was returned "
                "for this subquestion."
            )

        else:
            status = assessment.status
            strength = assessment.evidence_strength
            reason = assessment.reason.strip()

            # Discard invented or incorrectly assigned evidence IDs.
            evidence_ids = list(
                dict.fromkeys(
                    evidence_id
                    for evidence_id in assessment.evidence_ids
                    if evidence_id in accepted_evidence_ids
                )
            )

            if subquestion_id in duplicate_assessment_ids:
                status = SubquestionCoverageStatus.PARTIAL
                reason = (
                    "Duplicate coverage assessments were returned. "
                    + reason
                )

            elif (
                status == SubquestionCoverageStatus.SUPPORTED
                and not evidence_ids
            ):
                status = SubquestionCoverageStatus.PARTIAL
                reason = (
                    "No valid supporting evidence IDs were supplied. "
                    + reason
                )

            elif (
                status == SubquestionCoverageStatus.SUPPORTED
                and strength in {
                    EvidenceStrength.WEAK,
                    EvidenceStrength.UNKNOWN,
                }
            ):
                status = SubquestionCoverageStatus.PARTIAL
                reason = (
                    "Evidence strength is insufficient or unknown. "
                    + reason
                )

        # An explicit unresolved gap overrides a supported status.
        has_gap = any(
            gap.subquestion_id == subquestion_id
            for gap in gaps
        )

        if (
            has_gap
            and status == SubquestionCoverageStatus.SUPPORTED
        ):
            status = SubquestionCoverageStatus.PARTIAL
            reason = (
                "The evaluator reported an unresolved gap. "
                + reason
            )

        if status != SubquestionCoverageStatus.SUPPORTED:
            if not has_gap:
                gaps.append(
                    CoverageGap(
                        subquestion_id=subquestion_id,
                        description=reason,
                    )
                )

        normalized_assessments.append(
            SubquestionCoverage(
                subquestion_id=subquestion_id,
                status=status,
                evidence_strength=strength,
                evidence_ids=evidence_ids,
                reason=reason,
            )
        )

    # Guard against contradictory LLM output:
    # overall insufficient, but all objectives marked supported.
    if (
        coverage.status == CoverageStatus.INSUFFICIENT
        and not gaps
        and normalized_assessments
    ):
        first_assessment = normalized_assessments[0]

        reason = (
            "The evaluator reported insufficient overall coverage "
            "without identifying a specific gap."
        )

        normalized_assessments[0] = first_assessment.model_copy(
            update={
                "status": SubquestionCoverageStatus.PARTIAL,
                "reason": reason,
            }
        )

        gaps.append(
            CoverageGap(
                subquestion_id=first_assessment.subquestion_id,
                description=reason,
            )
        )

    all_supported = bool(normalized_assessments) and all(
        assessment.status == SubquestionCoverageStatus.SUPPORTED
        for assessment in normalized_assessments
    )

    is_sufficient = (
        coverage.status == CoverageStatus.SUFFICIENT
        and all_supported
        and not gaps
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

    logger.info(
        "Research coverage evaluation started.",
        extra={
            "event": "coverage_evaluation_started",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "research_coverage_evaluator",
            "iteration": iteration,
        },
    )

    if not subquestions:
        logger.error(
            "Research coverage evaluation has no subquestions."
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
            job_id=state["job_id"],
            research_session_id=state["research_session_id"],
            node="research_coverage_evaluator",
            iteration=iteration,
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
                "Research coverage evaluation failed."
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

        print('\n coverage eval response: ')
        print(coverage)

    logger.info(
        "Research coverage evaluation completed.",
        extra={
            "event": "coverage_evaluation_completed",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "research_coverage_evaluator",
            "iteration": iteration,
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
