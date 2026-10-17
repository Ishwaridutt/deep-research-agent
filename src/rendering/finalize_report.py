"""Final deterministic report rendering node."""

import logging

from src.graph.graph_state import WorkflowState
from src.models import ErrorType, ResearchError
from src.rendering.markdown import (
    ReportRenderingError,
    finalize_research_report,
)


logger = logging.getLogger(__name__)


def finalize_report_node(
    state: WorkflowState,
) -> dict:
    """Renders the validated ResearchReport into deterministic Markdown."""

    final_report = state["writer"]["final_report"]
    print("\nfinal report:", final_report)

    if final_report is None:
        error = ResearchError(
            error_type=ErrorType.INTERNAL,
            component="finalize_report",
            message=(
                "Final report rendering could not run because "
                "no validated ResearchReport exists."
            ),
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

    try:
        rendered_markdown = finalize_research_report(
            report=final_report,
            evidence_by_id=evidence_by_id,
            sources_by_id=sources_by_id,
            research_question=state["question"],
            stop_reason=state["research"]["stop_reason"],
        )

    except ReportRenderingError as exc:
        logger.exception(
            "Final report rendering failed.",
            extra={
                "event": "final_report_rendering_failed",
                "job_id": state["job_id"],
                "research_session_id": state["research_session_id"],
                "node": "finalize_report",
                "error_type": ErrorType.INTERNAL.value,
            },
        )

        error = ResearchError(
            error_type=ErrorType.INTERNAL,
            component="finalize_report",
            message=str(exc),
            recoverable=False,
        )

        return {
            "errors": [
                *state["errors"],
                error,
            ]
        }

    # Keep the structured ResearchReport and attach the rendered Markdown to it.
    final_report = final_report.model_copy(
        update={
            "markdown": rendered_markdown,
        }
    )

    logger.info(
        "Final report rendered.",
        extra={
            "event": "final_report_rendered",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "finalize_report",
        },
    )

    return {
        "writer": {
            **state["writer"],
            "final_report": final_report,
        }
    }

