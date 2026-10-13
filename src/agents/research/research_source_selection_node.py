"""Prepare job-scoped resources for the selected research sources."""

import logging

from langchain_core.runnables import RunnableConfig

from src.graph.graph_state import WorkflowState
from src.models import ErrorType, ResearchError, ResearchSource
from src.retrieval import RetrievalSessionFactory
from src.services.document_service import DocumentService
from src.services.research_job_resources import ResearchJobResources


logger = logging.getLogger(__name__)


def _selection_output(state: WorkflowState) -> dict:
    """Expose only fields needed for sanitized progress metadata."""

    return {
        "source_mode": state["source_mode"],
        "selected_document_ids": state["selected_document_ids"],
    }


def _get_job_resources(config: RunnableConfig) -> ResearchJobResources:
    """Read the resources created and ultimately cleared by ResearchService."""

    resources = config.get("configurable", {}).get("job_resources")
    if not isinstance(resources, ResearchJobResources):
        raise RuntimeError("Research job resources are unavailable.")
    return resources


async def research_source_selection_node(
    state: WorkflowState,
    config: RunnableConfig,
    *,
    document_service: DocumentService,
    retrieval_session_factory: RetrievalSessionFactory,
) -> dict:
    """Build one reusable private-document retrieval session when selected."""

    if ResearchSource.DOCUMENTS not in state["allowed_sources"]:
        return _selection_output(state)

    resources = _get_job_resources(config)
    if resources.document_retrieval_session is not None:
        return _selection_output(state)

    session = retrieval_session_factory.create_session()
    resources.document_retrieval_session = session

    try:
        await document_service.build_document_retrieval_session(
            owner_id=state["owner_id"] or "",
            document_ids=state["selected_document_ids"],
            session=session,
        )
    except ValueError as exc:
        resources.clear()
        error_type = ErrorType.VALIDATION
        message = str(exc)
    except Exception:
        logger.exception(
            "Private-document source preparation failed.",
            extra={
                "event": "document_source_preparation_failed",
                "job_id": state["job_id"],
                "research_session_id": state["research_session_id"],
                "node": "research_source_selection",
            },
        )
        resources.clear()
        error_type = ErrorType.INTERNAL
        message = "Private-document source preparation failed."
    else:
        logger.info(
            "Private-document source preparation completed.",
            extra={
                "event": "document_source_preparation_completed",
                "job_id": state["job_id"],
                "research_session_id": state["research_session_id"],
                "node": "research_source_selection",
                "document_count": len(state["selected_document_ids"]),
            },
        )
        return _selection_output(state)

    return {
        "errors": [
            *state["errors"],
            ResearchError(
                error_type=error_type,
                component="research_source_selection",
                message=message,
                recoverable=False,
            ),
        ],
    }
