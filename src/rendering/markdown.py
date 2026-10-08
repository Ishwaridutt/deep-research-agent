from __future__ import annotations

from typing import Any


class ReportRenderingError(RuntimeError):
    """Raised when the final report cannot be rendered."""


def finalize_research_report(
    *,
    report: Any,
    evidence_by_id: dict[str, Any],
    sources_by_id: dict[str, Any],
) -> str:
    """Render the final validated report into Markdown."""

    try:
        rendered = render_markdown_report(report)
    except Exception as exc:
        raise ReportRenderingError(
            f"Failed to render final report: {exc}"
        ) from exc

    if not rendered:
        raise ReportRenderingError(
            "Final report rendered as empty content."
        )

    return rendered


def render_markdown_report(report: Any) -> str:
    """Convert the final report payload into Markdown."""

    if report is None:
        return ""

    if isinstance(report, str):
        return report.strip()

    if hasattr(report, "model_dump"):
        report = report.model_dump()

    if isinstance(report, dict):
        return _render_dict_report(report)

    return str(report).strip()


def _render_dict_report(
    report: dict[str, Any],
) -> str:
    """Render a dictionary-based report as Markdown."""

    sections: list[str] = []

    title = report.get("title")
    if title:
        sections.append(f"# {title}")

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

    sources = report.get("sources")

    if sources:
        sources_markdown = _render_sources(
            sources
        )

        if sources_markdown:
            sections.append(
                "## Sources\n\n"
                + sources_markdown
            )

    if sections:
        return "\n\n".join(sections).strip()

    return str(report).strip()


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

    for source in sources:
        if isinstance(source, str):
            rendered_sources.append(
                f"- {source}"
            )
            continue

        if hasattr(source, "model_dump"):
            source = source.model_dump()

        if isinstance(source, dict):
            title = (
                source.get("title")
                or source.get("name")
                or source.get("url")
                or "Source"
            )

            url = source.get("url")

            if url:
                rendered_sources.append(
                    f"- [{title}]({url})"
                )
            else:
                rendered_sources.append(
                    f"- {title}"
                )

            continue

        rendered_sources.append(
            f"- {source}"
        )

    return "\n".join(rendered_sources)

