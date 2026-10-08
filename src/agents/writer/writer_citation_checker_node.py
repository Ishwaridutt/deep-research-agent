"""Writer Citation Checker node."""

import logging

from src.graph.graph_state import WorkflowState
from src.llm import (
    LLMCallMetadata,
    LLMRole,
    LLMService,
    LLMServiceError,
)
from src.models import (
    CitationCheckResult,
    CitationIssue,
    DraftReport,
    ErrorType,
    Evidence,
    ReportFinding,
    ReportSource,
    ResearchError,
    ResearchReport,
    SemanticCitationCheckOutput,
    Source,
)
from src.prompts.citation_checker_prompt import (
    build_citation_checker_messages,
)


logger = logging.getLogger(__name__)


def _validate_finding_references(
    *,
    draft_report: DraftReport,
    allowed_evidence_ids: set[str],
    evidence_by_id: dict[str, Evidence],
    source_ids: set[str],
) -> tuple[list[tuple[int, ReportFinding]], list[CitationIssue]]:
    """
    Performs deterministic citation-integrity checks.

    Only findings with valid evidence and source references are allowed
    to reach semantic citation verification.
    """

    valid_findings: list[tuple[int, ReportFinding]] = []
    issues: list[CitationIssue] = []

    for index, finding in enumerate(draft_report.findings):
        finding_issues: list[CitationIssue] = []

        if not finding.evidence_ids:
            finding_issues.append(
                CitationIssue(
                    finding_index=index,
                    description="Finding has no supporting evidence references.",
                )
            )

        for evidence_id in finding.evidence_ids:
            if evidence_id not in allowed_evidence_ids:
                finding_issues.append(
                    CitationIssue(
                        finding_index=index,
                        evidence_id=evidence_id,
                        description="Evidence was not supplied to the Writer Agent.",
                    )
                )
                continue

            evidence = evidence_by_id.get(evidence_id)

            if evidence is None:
                finding_issues.append(
                    CitationIssue(
                        finding_index=index,
                        evidence_id=evidence_id,
                        description="Referenced Evidence ID does not exist.",
                    )
                )
                continue

            if evidence.source_id not in source_ids:
                finding_issues.append(
                    CitationIssue(
                        finding_index=index,
                        evidence_id=evidence_id,
                        description="Referenced Evidence points to a Source that does not exist.",
                    )
                )

        if finding_issues:
            issues.extend(finding_issues)
            continue

        valid_findings.append((index, finding))

    return valid_findings, issues


def _collect_referenced_evidence(
    *,
    findings: list[tuple[int, ReportFinding]],
    evidence_by_id: dict[str, Evidence],
) -> list[Evidence]:
    """Returns unique Evidence objects referenced by candidate findings."""

    evidence_ids = dict.fromkeys(
        evidence_id
        for _, finding in findings
        for evidence_id in finding.evidence_ids
    )

    return [
        evidence_by_id[evidence_id]
        for evidence_id in evidence_ids
        if evidence_id in evidence_by_id
    ]


def _select_semantically_supported_findings(
    *,
    candidate_findings: list[tuple[int, ReportFinding]],
    semantic_output: SemanticCitationCheckOutput,
) -> tuple[list[ReportFinding], list[CitationIssue]]:
    """Keeps only findings confirmed as supported by semantic verification."""

    # Keep the first model decision if duplicate finding indexes are returned.
    semantic_results = {}

    for result in semantic_output.finding_results:
        semantic_results.setdefault(
            result.finding_index,
            (result.supported, result.reason),
        )

    supported_findings: list[ReportFinding] = []
    issues: list[CitationIssue] = []

    for finding_index, finding in candidate_findings:
        result = semantic_results.get(finding_index)

        if result is None:
            issues.append(
                CitationIssue(
                    finding_index=finding_index,
                    description=(
                        "Semantic citation checker did not return "
                        "a decision for this finding."
                    ),
                )
            )
            continue

        supported, reason = result

        if not supported:
            issues.append(
                CitationIssue(
                    finding_index=finding_index,
                    description=(
                        reason
                        or "Cited evidence did not sufficiently support the finding."
                    ),
                )
            )
            continue

        supported_findings.append(finding)

    return supported_findings, issues


def _build_report_sources(
    *,
    findings: list[ReportFinding],
    evidence_by_id: dict[str, Evidence],
    sources_by_id: dict[str, Source],
) -> list[ReportSource]:
    """Builds source metadata used by the final supported findings."""

    report_sources: list[ReportSource] = []
    seen_source_ids: set[str] = set()

    for finding in findings:
        for evidence_id in finding.evidence_ids:
            evidence = evidence_by_id.get(evidence_id)

            if evidence is None:
                continue

            source = sources_by_id.get(evidence.source_id)

            if source is None or source.id in seen_source_ids:
                continue

            seen_source_ids.add(source.id)

            report_sources.append(
                ReportSource(
                    source_id=source.id,
                    title=source.title or str(source.uri),
                    uri=str(source.uri),
                )
            )

    return report_sources


def _with_citation_limitation(
    limitations: list[str],
    *,
    issues: list[CitationIssue],
) -> list[str]:
    """Adds one user-facing limitation when citation problems removed findings."""

    updated = list(limitations)

    if not issues:
        return updated

    message = (
        "One or more draft findings were removed because citation verification "
        "could not confirm sufficient evidence support."
    )

    if message not in updated:
        updated.append(message)

    return updated


def _build_no_supported_findings_report(
    *,
    draft_report: DraftReport,
    issues: list[CitationIssue],
) -> ResearchReport:
    """Builds a fail-closed report when no finding survives citation validation."""

    return ResearchReport(
        executive_summary=(
            "The available research evidence did not yield any findings that "
            "could be included in the final report after citation validation."
        ),
        findings=[],
        limitations=_with_citation_limitation(
            draft_report.limitations,
            issues=issues,
        ),
        sources=[],
    )


async def writer_citation_checker_node(
    state: WorkflowState,
    *,
    llm_service: LLMService,
) -> dict:
    """
    Validates DraftReport citations and creates the final ResearchReport.

    Validation uses deterministic reference checks followed by semantic
    evidence-to-claim verification. Unsupported findings are removed.
    """

    iteration = state["research"]["iteration_count"]

    logger.info(
        "Writer citation checking started.",
        extra={
            "event": "writer_citation_check_started",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "writer_citation_checker",
            "iteration": iteration,
        },
    )

    draft_report = state["writer"]["draft_report"]

    if draft_report is None:
        error = ResearchError(
            error_type=ErrorType.CITATION,
            component="writer_citation_checker",
            message="Citation checking could not run because no DraftReport exists.",
            recoverable=False,
        )

        return {
            "errors": [
                *state["errors"],
                error,
            ]
        }

    evidence_by_id = {
        evidence.id: evidence
        for evidence in state["research"]["evidence"]
    }

    sources_by_id = {
        source.id: source
        for source in state["research"]["sources"]
    }

    candidate_findings, deterministic_issues = _validate_finding_references(
        draft_report=draft_report,
        allowed_evidence_ids=set(state["writer"]["evidence_ids"]),
        evidence_by_id=evidence_by_id,
        source_ids=set(sources_by_id),
    )

    # Avoid an LLM call when deterministic checks already removed every finding.
    if not candidate_findings:
        citation_result = CitationCheckResult(
            valid=not deterministic_issues,
            issues=deterministic_issues,
        )

        final_report = _build_no_supported_findings_report(
            draft_report=draft_report,
            issues=deterministic_issues,
        )

        return {
            "writer": {
                **state["writer"],
                "citation_result": citation_result,
                "final_report": final_report,
            }
        }

    referenced_evidence = _collect_referenced_evidence(
        findings=candidate_findings,
        evidence_by_id=evidence_by_id,
    )

    messages = build_citation_checker_messages(
        original_question=state["question"],
        findings=candidate_findings,
        evidence=referenced_evidence,
    )

    metadata = LLMCallMetadata(
        job_id=state["job_id"],
        research_session_id=state["research_session_id"],
        node="writer_citation_checker",
        iteration=iteration,
    )

    try:
        semantic_output = await llm_service.invoke_structured(
            role=LLMRole.CITATION_CHECKER,
            messages=messages,
            output_schema=SemanticCitationCheckOutput,
            metadata=metadata,
        )

    except LLMServiceError as exc:
        logger.exception(
            "Semantic citation checking failed.",
            extra={
                "event": "writer_citation_check_failed",
                "job_id": state["job_id"],
                "research_session_id": state["research_session_id"],
                "node": "writer_citation_checker",
                "iteration": iteration,
                "error_type": ErrorType.CITATION.value,
            },
        )

        error = ResearchError(
            error_type=ErrorType.CITATION,
            component="writer_citation_checker",
            message=str(exc),
            recoverable=False,
        )

        return {
            "errors": [
                *state["errors"],
                error,
            ]
        }

    supported_findings, semantic_issues = _select_semantically_supported_findings(
        candidate_findings=candidate_findings,
        semantic_output=semantic_output,
    )

    all_issues = [
        *deterministic_issues,
        *semantic_issues,
    ]

    citation_result = CitationCheckResult(
        valid=not all_issues,
        issues=all_issues,
    )

    if not supported_findings:
        final_report = _build_no_supported_findings_report(
            draft_report=draft_report,
            issues=all_issues,
        )

    else:
        final_report = ResearchReport(
            executive_summary=semantic_output.revised_executive_summary.strip(),
            findings=supported_findings,
            limitations=_with_citation_limitation(
                draft_report.limitations,
                issues=all_issues,
            ),
            sources=_build_report_sources(
                findings=supported_findings,
                evidence_by_id=evidence_by_id,
                sources_by_id=sources_by_id,
            ),
        )

    logger.info(
        "Writer citation checking completed.",
        extra={
            "event": "writer_citation_check_completed",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "writer_citation_checker",
            "iteration": iteration,
        },
    )

    return {
        "writer": {
            **state["writer"],
            "citation_result": citation_result,
            "final_report": final_report,
        }
    }

    