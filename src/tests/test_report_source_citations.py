"""Focused tests for user-facing report citations and Markdown."""

from datetime import UTC, datetime

from src.agents.writer.writer_citation_checker_node import (
    _build_report_sources,
    _build_user_facing_report_content,
    _extract_evidence_ids,
)
from src.models import (
    Evidence,
    ReportFinding,
    ResearchReport,
    ResearchStopReason,
    Source,
    SourceProvider,
    SourceType,
)
from src.rendering.markdown import render_markdown_report


def _evidence(evidence_id: str, source_id: str) -> Evidence:
    return Evidence(
        id=evidence_id,
        subquestion_id="SQ-1",
        source_id=source_id,
        chunk_id=f"{source_id}:0",
        source_type=SourceType.WEB,
        content="Supporting evidence.",
        relevance_score=0.9,
    )


def _source(source_id: str, title: str) -> Source:
    return Source(
        id=source_id,
        source_type=SourceType.WEB,
        provider=SourceProvider.TAVILY,
        canonical_id=f"url:https://example.com/{source_id}",
        title=title,
        uri=f"https://example.com/{source_id}",
        content="Source content.",
        retrieved_at=datetime.now(UTC),
    )


def test_final_report_uses_source_numbers_and_hides_evidence_ids():
    evidence = {
        "EV-1": _evidence("EV-1", "SRC-1"),
        "EV-2": _evidence("EV-2", "SRC-1"),
        "EV-3": _evidence("EV-3", "SRC-2"),
    }
    findings = [
        ReportFinding(
            title="Finding",
            content="First claim. [EV-1][EV-2] Second claim. [EV-3]",
            evidence_ids=["EV-1", "EV-2", "EV-3"],
        )
    ]
    limitations = ["A limitation. [EV-3]"]
    sources = _build_report_sources(
        findings=findings,
        evidence_by_id=evidence,
        sources_by_id={
            "SRC-1": _source("SRC-1", "Source one"),
            "SRC-2": _source("SRC-2", "Source two"),
        },
        additional_evidence_ids=_extract_evidence_ids(limitations),
    )

    final_findings, final_limitations = _build_user_facing_report_content(
        findings=findings,
        limitations=limitations,
        sources=sources,
        evidence_by_id=evidence,
    )
    report = ResearchReport(
        executive_summary="Summary.",
        findings=final_findings,
        limitations=final_limitations,
        sources=sources,
    )
    payload = report.model_dump(mode="json")

    assert payload["findings"] == [{
        "section": "Detailed Analysis",
        "title": "Finding",
        "content": "First claim. [1] Second claim. [2]",
    }]
    assert payload["limitations"] == ["A limitation. [2]"]
    assert "EV-" not in str(payload)


def test_markdown_contains_full_report_and_numbered_linked_sources():
    evidence = {"EV-1": _evidence("EV-1", "SRC-1")}
    findings = [
        ReportFinding(
            title="Finding",
            content="Supported claim. [EV-1]",
            evidence_ids=["EV-1"],
        )
    ]
    sources = _build_report_sources(
        findings=findings,
        evidence_by_id=evidence,
        sources_by_id={"SRC-1": _source("SRC-1", "Source one")},
    )
    final_findings, _ = _build_user_facing_report_content(
        findings=findings,
        limitations=[],
        sources=sources,
        evidence_by_id=evidence,
    )
    markdown = render_markdown_report(
        ResearchReport(
            executive_summary="Summary.",
            findings=final_findings,
            sources=sources,
        ),
        research_question="What happened?",
        stop_reason=ResearchStopReason.MAX_ITERATIONS,
    )

    assert markdown.startswith("# Deep Research Report")
    assert "## Research Question\n\nWhat happened?" in markdown
    assert "## Executive Summary\n\nSummary." in markdown
    assert "## Research Approach" in markdown
    assert "1 citation-validated finding from 1 referenced source" in markdown
    assert "research-round limit was reached" in markdown
    assert "## Detailed Analysis" in markdown
    assert "### Finding\n\nSupported claim. [1]" in markdown
    assert "1. [Source one](https://example.com/SRC-1)" in markdown
    assert "EV-" not in markdown


def test_sufficient_coverage_does_not_claim_every_objective_was_perfectly_covered():
    markdown = render_markdown_report(
        ResearchReport(executive_summary="A qualified answer.", limitations=["Minor background gap."]),
        stop_reason=ResearchStopReason.COVERAGE_SUFFICIENT,
    )
    assert "sufficient for a grounded report on the core question" in markdown
    assert "## Evidence Gaps and Limitations\n\n- Minor background gap." in markdown
    assert "planned evidence coverage was satisfied" not in markdown
