from __future__ import annotations

from typing import Any


class ReportRenderingError(RuntimeError):
    """Raised when the final report cannot be rendered."""


def finalize_research_report(
    *,
    report: Any,
    evidence_by_id: dict[str, Any],
    sources_by_id: dict[str, Any],
    research_question: str | None = None,
    stop_reason: Any = None,
) -> str:
    """Render the final validated report into Markdown."""

    try:
        rendered = render_markdown_report(
            report,
            research_question=research_question,
            stop_reason=stop_reason,
        )
    except Exception as exc:
        raise ReportRenderingError(
            f"Failed to render final report: {exc}"
        ) from exc

    if not rendered:
        raise ReportRenderingError(
            "Final report rendered as empty content."
        )

    return rendered


def render_markdown_report(
    report: Any,
    *,
    research_question: str | None = None,
    stop_reason: Any = None,
) -> str:
    """Convert the final report payload into Markdown."""

    if report is None:
        return ""

    if isinstance(report, str):
        return report.strip()

    if hasattr(report, "model_dump"):
        report = report.model_dump()

    if isinstance(report, dict):
        return _render_dict_report(
            report,
            research_question=research_question,
            stop_reason=stop_reason,
        )

    return str(report).strip()


def _render_dict_report(
    report: dict[str, Any],
    *,
    research_question: str | None = None,
    stop_reason: Any = None,
) -> str:
    """Render a dictionary-based report as Markdown."""

    title = report.get("title") or "Deep Research Report"
    sections: list[str] = [f"# {title}"]

    if research_question:
        sections.append(
            "## Research Question\n\n"
            + research_question.strip()
        )

    executive_summary = report.get("executive_summary")
    if executive_summary:
        sections.append(
            "## Executive Summary\n\n"
            + str(executive_summary).strip()
        )

    findings = report.get("findings")
    if findings:
        grouped_findings: dict[str, list[str]] = {}
        for finding in findings:
            if hasattr(finding, "model_dump"):
                finding = finding.model_dump()
            if not isinstance(finding, dict):
                continue
            heading = str(finding.get("section") or "Detailed Analysis").strip() or "Detailed Analysis"
            grouped_findings.setdefault(heading, []).append(
                f"### {finding.get('title', 'Finding')}\n\n"
                f"{str(finding.get('content', '')).strip()}"
            )
        sections.extend(
            f"## {heading}\n\n" + "\n\n".join(items)
            for heading, items in grouped_findings.items()
        )

    limitations = report.get("limitations")
    if limitations:
        sections.append(
            "## Evidence Gaps and Limitations\n\n"
            + "\n".join(f"- {item}" for item in limitations)
        )

    approach = _render_research_approach(
        finding_count=len(findings or []),
        source_count=len(report.get("sources") or []),
        stop_reason=stop_reason,
    )
    sections.insert(
        3 if research_question else 2,
        "## Research Approach\n\n" + approach,
    )

    summary = report.get("summary")
    if summary:
        sections.append(str(summary).strip())

    content = (
        report.get("content")
        or report.get("report")
        or report.get("final_report")
        or report.get("body")
    )

    if content:
        sections.append(str(content).strip())

    unverified = _render_unverified_material(report.get("unverified_material") or [])
    if unverified:
        sections.append(unverified)

    sources = report.get("sources")

    if sources:
        sources_markdown = _render_sources(sources)

        if sources_markdown:
            sections.append(
                "## Cited Sources\n\n"
                + sources_markdown
            )

    if sections:
        return "\n\n".join(sections).strip()

    return str(report).strip()


def _render_unverified_material(items: list[Any]) -> str:
    """Render draft content that citation checking could not verify, clearly labelled."""

    blocks: list[str] = []
    for item in items:
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        if not isinstance(item, dict) or not str(item.get("content") or "").strip():
            continue
        links = []
        for source in item.get("sources") or []:
            if hasattr(source, "model_dump"):
                source = source.model_dump()
            title = str(source.get("title") or "Source").strip()
            uri = str(source.get("uri") or "")
            links.append(f"[{title}]({uri})" if uri.startswith(("http://", "https://")) else title)
        block = f"### {item.get('title') or 'Unverified statement'}\n\n{str(item['content']).strip()}"
        if links:
            block += "\n\nPossible sources: " + "; ".join(links)
        blocks.append(block)

    if not blocks:
        return ""
    return (
        "## Additional material (not citation-verified)\n\n"
        "_These statements could not be verified against their cited evidence. "
        "They may still be useful; check the linked sources before relying on them._\n\n"
        + "\n\n".join(blocks)
    )


def _render_research_approach(
    *,
    finding_count: int,
    source_count: int,
    stop_reason: Any,
) -> str:
    """Summarize report scope and the reason the research loop stopped."""

    finding_label = "finding" if finding_count == 1 else "findings"
    source_label = "source" if source_count == 1 else "sources"
    text = (
        f"This report synthesizes {finding_count} citation-validated {finding_label} "
        f"from {source_count} referenced {source_label}."
    )
    reason = getattr(stop_reason, "value", stop_reason)
    status_messages = {
        "coverage_sufficient": "The evidence was judged sufficient for a grounded report on the core question.",
        "max_iterations": (
            "The configured research-round limit was reached; unresolved evidence "
            "gaps are disclosed below."
        ),
        "query_budget_exhausted": (
            "The search-query budget was reached; unresolved evidence gaps are "
            "disclosed below."
        ),
        "extraction_budget_exhausted": (
            "The source-extraction budget was reached; unresolved evidence gaps "
            "are disclosed below."
        ),
        "time_budget_exhausted": (
            "The research time budget was reached; unresolved evidence gaps are "
            "disclosed below."
        ),
        "no_evidence_improvement": (
            "Research stopped after an additional round produced no meaningful "
            "evidence improvement."
        ),
        "no_actionable_gaps": (
            "Research stopped because no unresolved gap had an actionable search target."
        ),
    }
    status = status_messages.get(reason)
    return f"{text} {status}" if status else text


def _render_sources(
    sources: Any,
) -> str:
    """Render source information into a Markdown list."""

    if isinstance(sources, str):
        return sources.strip()

    if not isinstance(
        sources,
        (list, tuple),
    ):
        return str(sources).strip()

    rendered_sources: list[str] = []

    for index, source in enumerate(sources, start=1):
        if isinstance(source, str):
            rendered_sources.append(
                f"{index}. {source}"
            )
            continue

        if hasattr(source, "model_dump"):
            source = source.model_dump()

        if isinstance(source, dict):
            title = (
                source.get("title")
                or source.get("name")
                or source.get("uri")
                or source.get("url")
                or "Source"
            )

            url = source.get("uri") or source.get("url")

            if isinstance(url, str) and url.startswith(("http://", "https://")):
                rendered_sources.append(
                    f"{index}. [{title}]({url})"
                )
            else:
                rendered_sources.append(
                    f"{index}. {title}"
                )

            continue

        rendered_sources.append(
            f"{index}. {source}"
        )

    return "\n".join(rendered_sources)

