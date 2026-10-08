"""Research Coverage Evaluator node."""

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
    CoverageResult,
    CoverageStatus,
    ErrorType,
    Evidence,
    ResearchError,
    SubQuestion,
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
    """Groups accepted evidence by its owning subquestion."""

    grouped_evidence = {
        subquestion.id: []
        for subquestion in subquestions
    }

    for evidence_item in evidence:
        if evidence_item.subquestion_id in grouped_evidence:
            grouped_evidence[evidence_item.subquestion_id].append(
                evidence_item
            )

    return grouped_evidence


def _build_missing_evidence_coverage(
    *,
    subquestions: list[SubQuestion],
    evidence_by_subquestion: dict[str, list[Evidence]],
) -> CoverageResult:
    """Builds insufficient coverage for subquestions with no evidence."""

    return CoverageResult(
        status=CoverageStatus.INSUFFICIENT,
        gaps=[
            CoverageGap(
                subquestion_id=subquestion.id,
                description=(
                    "No usable evidence has been collected "
                    "for this subquestion."
                ),
            )
            for subquestion in subquestions
            if not evidence_by_subquestion[subquestion.id]
        ],
    )


def _normalize_coverage_result(
    *,
    coverage: CoverageResult,
    subquestions: list[SubQuestion],
) -> CoverageResult:
    """Normalizes LLM coverage output before storing it."""

    if coverage.status == CoverageStatus.SUFFICIENT:
        return CoverageResult(
            status=CoverageStatus.SUFFICIENT,
            gaps=[],
        )

    valid_subquestion_ids = {
        subquestion.id
        for subquestion in subquestions
    }

    gap_descriptions: dict[str, list[str]] = {}

    # Remove invalid IDs, empty descriptions, and duplicate gaps.
    for gap in coverage.gaps:
        if gap.subquestion_id not in valid_subquestion_ids:
            continue

        description = gap.description.strip()

        if not description:
            continue

        descriptions = gap_descriptions.setdefault(
            gap.subquestion_id,
            [],
        )

        if description not in descriptions:
            descriptions.append(description)

    if gap_descriptions:
        return CoverageResult(
            status=CoverageStatus.INSUFFICIENT,
            gaps=[
                CoverageGap(
                    subquestion_id=subquestion_id,
                    description=" ".join(descriptions),
                )
                for subquestion_id, descriptions
                in gap_descriptions.items()
            ],
        )

    # If the model declares insufficient coverage but gives no usable gap,
    # allow every subquestion to participate in another research iteration.
    return CoverageResult(
        status=CoverageStatus.INSUFFICIENT,
        gaps=[
            CoverageGap(
                subquestion_id=subquestion.id,
                description=(
                    "Additional evidence is needed to establish "
                    "sufficient coverage for this subquestion."
                ),
            )
            for subquestion in subquestions
        ],
    )


async def research_coverage_evaluator_node(
    state: WorkflowState,
    *,
    llm_service: LLMService,
) -> dict:
    """
    Evaluates whether accumulated evidence sufficiently covers the research plan.

    Running this node marks the current research iteration as completed.
    """

    completed_iteration = (
        state["research"]["iteration_count"] + 1
    )

    subquestions = state["research"]["subquestions"]
    evidence = state["research"]["evidence"]
    errors = list(state["errors"])

    logger.info(
        "Research coverage evaluation started.",
        extra={
            "event": "coverage_evaluation_started",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "research_coverage_evaluator",
            "iteration": completed_iteration,
        },
    )

    evidence_by_subquestion = _group_evidence_by_subquestion(
        subquestions=subquestions,
        evidence=evidence,
    )

    # A research plan cannot have sufficient coverage while any planned
    # subquestion has no accepted evidence, so avoid an unnecessary LLM call.
    if any(
        not evidence_by_subquestion[subquestion.id]
        for subquestion in subquestions
    ):
        coverage = _build_missing_evidence_coverage(
            subquestions=subquestions,
            evidence_by_subquestion=evidence_by_subquestion,
        )

        logger.warning(
            "Research iteration completed with missing evidence.",
            extra={
                "event": "coverage_missing_evidence",
                "job_id": state["job_id"],
                "research_session_id": state["research_session_id"],
                "node": "research_coverage_evaluator",
                "iteration": completed_iteration,
            },
        )

        return {
            "research": {
                **state["research"],
                "iteration_count": completed_iteration,
                "coverage": coverage,
            },
            "errors": errors,
        }

    coverage_evaluator_prompt = build_coverage_evaluator_messages(
        original_question=state["question"],
        subquestions=subquestions,
        evidence_by_subquestion=evidence_by_subquestion,
    )

    metadata = LLMCallMetadata(
        job_id=state["job_id"],
        research_session_id=state["research_session_id"],
        node="research_coverage_evaluator",
        iteration=completed_iteration,
    )

    try:
        raw_coverage = await llm_service.invoke_structured(
            role=LLMRole.COVERAGE_EVALUATOR,
            messages=coverage_evaluator_prompt,
            output_schema=CoverageResult,
            metadata=metadata,
        )

        print('\nCoverage eval LLM resp raw_coverage: ')
        print(raw_coverage)

    except LLMServiceError as exc:
        logger.exception(
            "Research coverage evaluation failed.",
            extra={
                "event": "coverage_evaluation_failed",
                "job_id": state["job_id"],
                "research_session_id": state["research_session_id"],
                "node": "research_coverage_evaluator",
                "iteration": completed_iteration,
                "error_type": ErrorType.COVERAGE.value,
            },
        )

        errors.append(
            ResearchError(
                error_type=ErrorType.COVERAGE,
                component="research_coverage_evaluator",
                message=str(exc),
                recoverable=False,
            )
        )

        return {
            "research": {
                **state["research"],
                "iteration_count": completed_iteration,
            },
            "errors": errors,
        }

    coverage = _normalize_coverage_result(
        coverage=raw_coverage,
        subquestions=subquestions,
    )

    logger.info(
        "Research coverage evaluation completed.",
        extra={
            "event": "coverage_evaluation_completed",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "research_coverage_evaluator",
            "iteration": completed_iteration,
        },
    )

    return {
        "research": {
            **state["research"],
            "iteration_count": completed_iteration,
            "coverage": coverage,
        },
        "errors": errors,
    }

