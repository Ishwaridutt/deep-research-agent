"""Application service for executing deep research workflows."""

import logging
from typing import cast

from langgraph.graph.state import CompiledStateGraph

from src.config import Settings
from src.graph.graph_state import WorkflowState, create_initial_workflow_state
from src.models import (
    ErrorType,
    ResearchError,
    ResearchExecutionResult,
    ResearchExecutionStatus,
    ResearchSource,
)
from src.services.generate_research_ids import (
    generate_job_id,
    generate_research_session_id,
)


logger = logging.getLogger(__name__)


class ResearchService:
    """
    Application-level entry point for running a research workflow.

    HTTP routes depend on this service rather than invoking LangGraph,
    constructing workflow state, or generating IDs themselves.
    """

    def __init__(
        self,
        *,
        graph: CompiledStateGraph,
        settings: Settings,
    ) -> None:
        """Stores the compiled workflow and application configuration."""

        self._graph = graph
        self._settings = settings

    async def research(
        self,
        *,
        question: str,
        allowed_sources: list[ResearchSource] | None = None,
        research_session_id: str | None = None,
    ) -> ResearchExecutionResult:
        """
        Executes one complete research job.

        A new job ID is always created. A research session ID is reused when
        supplied, otherwise a new one is generated.
        """

        job_id = generate_job_id()

        supplied_session_id = (
            research_session_id.strip()
            if research_session_id
            else ""
        )

        session_id = (
            supplied_session_id
            or generate_research_session_id()
        )

        sources = (
            list(allowed_sources)
            if allowed_sources is not None
            else [ResearchSource.WEB]
        )

        initial_state = create_initial_workflow_state(
            job_id=job_id,
            research_session_id=session_id,
            question=question,
            allowed_sources=sources,
        )

        logger.info(
            "Research workflow execution started.",
            extra={
                "event": "research_workflow_started",
                "job_id": job_id,
                "research_session_id": session_id,
                "node": "research_service",
            },
        )

        config = {
            "run_name": "deep_research_job",
            "tags": [
                "deep-research-agent",
                "research-job",
            ],
            "metadata": {
                "job_id": job_id,
                "research_session_id": session_id,
            },
            "recursion_limit": self._settings.app.graph_recursion_limit,
        }

        try:
            graph_result = await self._graph.ainvoke(
                initial_state,
                config=config,
            )

        except Exception as exc:
            logger.exception(
                "Research workflow execution failed unexpectedly.",
                extra={
                    "event": "research_workflow_failed",
                    "job_id": job_id,
                    "research_session_id": session_id,
                    "node": "research_service",
                    "error_type": ErrorType.INTERNAL.value,
                },
            )

            error = ResearchError(
                error_type=ErrorType.INTERNAL,
                component="research_service",
                message=f"The research workflow failed unexpectedly: {exc}",
                recoverable=False,
            )

            return ResearchExecutionResult(
                job_id=job_id,
                research_session_id=session_id,
                status=ResearchExecutionStatus.FAILED,
                stop_reason=None,
                report=None,
                errors=[error],
            )

        final_state = cast(
            WorkflowState,
            graph_result,
        )

        return self._build_result(
            state=final_state,
        )

    def _build_result(
        self,
        *,
        state: WorkflowState,
    ) -> ResearchExecutionResult:
        """Converts final LangGraph state into an application-level result."""

        errors = list(state["errors"])
        final_report = state["writer"]["final_report"]

        terminal_errors = [
            error
            for error in errors
            if not error.recoverable
        ]

        # A completed graph without a final report is an invalid terminal state.
        if final_report is None and not terminal_errors:
            errors.append(
                ResearchError(
                    error_type=ErrorType.INTERNAL,
                    component="research_service",
                    message=(
                        "Research workflow completed without producing "
                        "a final report."
                    ),
                    recoverable=False,
                )
            )

            terminal_errors = [
                error
                for error in errors
                if not error.recoverable
            ]

        if terminal_errors:
            status = ResearchExecutionStatus.FAILED
            report = None

        else:
            status = ResearchExecutionStatus.COMPLETED
            report = final_report

        logger.info(
            "Research workflow execution finished.",
            extra={
                "event": "research_workflow_finished",
                "job_id": state["job_id"],
                "research_session_id": state["research_session_id"],
                "node": "research_service",
                "iteration": state["research"]["iteration_count"],
            },
        )

        return ResearchExecutionResult(
            job_id=state["job_id"],
            research_session_id=state["research_session_id"],
            status=status,
            stop_reason=state["research"]["stop_reason"],
            report=report,
            errors=errors,
        )

    