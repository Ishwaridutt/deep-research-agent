"""Application service for synchronous and persisted research workflows."""

import asyncio
import logging
from typing import cast

from langgraph.graph.state import CompiledStateGraph

from src.config import Settings
from src.graph.graph_state import WorkflowState, create_initial_workflow_state
from src.llm import LLMService
from src.models import (
    ErrorType,
    ResearchError,
    ResearchExecutionResult,
    ResearchExecutionStatus,
    ResearchSource,
    ResearchSourceMode,
    ResearchSourceSelection,
)
from src.models.pipeline_event import ResearchJobCreated
from src.persistence import ResearchJobStore
from src.services.generate_research_ids import (
    generate_job_id,
    generate_research_session_id,
)
from src.services.research_pipeline_stream_service import (
    PipelineRecorder,
    run_graph_with_events,
)
from src.services.research_job_resources import ResearchJobResources
from src.services.document_service import DocumentService


logger = logging.getLogger(__name__)


class ResearchJobQueueFull(Exception):
    """The local execution queue has reached its configured limit."""


class ResearchRequestValidationError(ValueError):
    """Raised when a request cannot safely enter the research graph."""


def _failure_details(result: ResearchExecutionResult) -> dict:
    """Build a sanitized summary from errors already safe for the public API."""

    terminal_errors = [error for error in result.errors if not error.recoverable]
    errors = terminal_errors or result.errors
    if not errors:
        return {}
    error = errors[0]
    return {
        "error_type": error.error_type.value,
        "component": error.component,
        "reason": error.message,
        "error_count": len(errors),
    }


class ResearchService:
    """Owns research execution; HTTP handlers do not invoke LangGraph."""

    def __init__(
        self,
        *,
        graph: CompiledStateGraph,
        settings: Settings,
        job_store: ResearchJobStore,
        llm_service: LLMService | None = None,
        document_service: DocumentService | None = None,
        max_concurrent_jobs: int = 2,
        max_pending_jobs: int = 20,
    ) -> None:
        self._graph = graph
        self._settings = settings
        self._job_store = job_store
        self._llm_service = llm_service
        self._document_service = document_service
        self._task_slots = asyncio.Semaphore(max_concurrent_jobs)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._admission_lock = asyncio.Lock()
        self._max_pending_jobs = max_pending_jobs
        self._shutting_down = False

    async def _validate_source_selection(
        self,
        *,
        source_selection: ResearchSourceSelection | None,
        owner_id: str | None,
    ) -> None:
        """Validate private-document availability before spending LLM budget."""

        selection = source_selection or ResearchSourceSelection()
        if selection.mode == ResearchSourceMode.WEB:
            return

        normalized_owner_id = (owner_id or "").strip()
        if not normalized_owner_id:
            raise ResearchRequestValidationError(
                "X-Owner-Id is required when private documents are selected."
            )
        if not selection.document_ids:
            raise ResearchRequestValidationError(
                "At least one private document must be selected for document research."
            )
        if self._document_service is None:
            raise ResearchRequestValidationError(
                "Private-document validation is unavailable."
            )

        try:
            await self._document_service.validate_documents_available(
                normalized_owner_id,
                selection.document_ids,
            )
        except ValueError as exc:
            raise ResearchRequestValidationError(str(exc)) from exc

    def _create_state(
        self,
        *,
        question: str,
        source_selection: ResearchSourceSelection | None,
        owner_id: str | None,
        research_session_id: str | None,
    ) -> WorkflowState:
        selection = source_selection or ResearchSourceSelection()
        sources = {
            ResearchSourceMode.WEB: [ResearchSource.WEB],
            ResearchSourceMode.DOCUMENTS: [ResearchSource.DOCUMENTS],
            ResearchSourceMode.HYBRID: [
                ResearchSource.WEB,
                ResearchSource.DOCUMENTS,
            ],
        }[selection.mode]
        return create_initial_workflow_state(
            job_id=generate_job_id(),
            research_session_id=(research_session_id or "").strip() or generate_research_session_id(),
            question=question,
            allowed_sources=sources,
            source_mode=selection.mode,
            owner_id=(owner_id or "").strip() or None,
            selected_document_ids=list(dict.fromkeys(selection.document_ids)),
        )

    def _graph_config(
        self,
        state: WorkflowState,
        resources: ResearchJobResources,
    ) -> dict:
        return {
            "run_name": "deep_research_job",
            "tags": ["deep-research-agent", "research-job"],
            "metadata": {
                "job_id": state["job_id"],
                "research_session_id": state["research_session_id"],
            },
            "configurable": {
                "job_resources": resources,
            },
            "recursion_limit": self._settings.app.graph_recursion_limit,
        }

    def _failed_result(self, state: WorkflowState, message: str) -> ResearchExecutionResult:
        return ResearchExecutionResult(
            job_id=state["job_id"],
            research_session_id=state["research_session_id"],
            status=ResearchExecutionStatus.FAILED,
            stop_reason=None,
            report=None,
            errors=[ResearchError(
                error_type=ErrorType.INTERNAL,
                component="research_service",
                message=message,
                recoverable=False,
            )],
        )

    async def research(
        self,
        *,
        question: str,
        source_selection: ResearchSourceSelection | None = None,
        owner_id: str | None = None,
        research_session_id: str | None = None,
    ) -> ResearchExecutionResult:
        """Preserve the existing synchronous /api/research behavior."""

        await self._validate_source_selection(
            source_selection=source_selection,
            owner_id=owner_id,
        )
        state = self._create_state(
            question=question,
            source_selection=source_selection,
            owner_id=owner_id,
            research_session_id=research_session_id,
        )
        resources = ResearchJobResources()
        logger.info(
            "Research workflow execution started.",
            extra={
                "event": "research_workflow_started",
                "job_id": state["job_id"],
                "research_session_id": state["research_session_id"],
                "node": "research_service",
            },
        )
        try:
            result = await self._graph.ainvoke(
                state,
                config=self._graph_config(state, resources),
            )
        except Exception as exc:
            logger.exception("Research workflow execution failed unexpectedly.")
            return self._failed_result(state, f"The research workflow failed unexpectedly: {exc}")
        finally:
            resources.clear()
            if self._llm_service is not None:
                self._llm_service.release_job(state["job_id"])
        return self._build_result(state=cast(WorkflowState, result))

    async def start_job(
        self,
        *,
        question: str,
        source_selection: ResearchSourceSelection | None = None,
        owner_id: str | None = None,
        research_session_id: str | None = None,
    ) -> ResearchJobCreated:
        """Persist a queued job and start execution independently of SSE."""

        await self._validate_source_selection(
            source_selection=source_selection,
            owner_id=owner_id,
        )
        async with self._admission_lock:
            if self._shutting_down or len(self._tasks) >= self._max_pending_jobs:
                raise ResearchJobQueueFull("The research job queue is currently full.")

            state = self._create_state(
                question=question,
                source_selection=source_selection,
                owner_id=owner_id,
                research_session_id=research_session_id,
            )
            await self._job_store.create_job(
                job_id=state["job_id"],
                research_session_id=state["research_session_id"],
                question=question,
                allowed_sources=[source.value for source in state["allowed_sources"]],
            )

            task = asyncio.create_task(self._run_persisted_job(state))
            self._tasks[state["job_id"]] = task
            task.add_done_callback(
                lambda finished, job_id=state["job_id"]: self._on_job_finished(job_id, finished)
            )
        return ResearchJobCreated(
            job_id=state["job_id"], research_session_id=state["research_session_id"],
        )

    def _on_job_finished(self, job_id: str, task: asyncio.Task[None]) -> None:
        """Release the job slot and surface unexpected task-level errors."""

        self._tasks.pop(job_id, None)
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.error(
                    "Research job task terminated unexpectedly.",
                    extra={"job_id": job_id, "event": "research_job_task_failed"},
                    exc_info=(type(error), error, error.__traceback__),
                )

    async def _run_persisted_job(self, state: WorkflowState) -> None:
        """Run the graph once, commit events, and atomically commit the result."""

        recorder = PipelineRecorder(
            job_id=state["job_id"],
            persist=self._job_store.append_event,
        )
        failure_message: str | None = None
        resources = ResearchJobResources()

        try:
            try:
                async with self._task_slots:
                    await self._job_store.mark_running(state["job_id"])
                    final_state = await run_graph_with_events(
                        graph=self._graph,
                        state=state,
                        config=self._graph_config(state, resources),
                        recorder=recorder,
                    )
                    result = self._build_result(state=final_state)
            except asyncio.CancelledError:
                failure_message = "Research job stopped during server shutdown."
                result = self._failed_result(state, failure_message)
                # Persist terminal status before propagating task cancellation.
                await self._finish_persisted_job(
                    recorder,
                    result,
                    failure_message,
                )
                raise
            except Exception:
                logger.exception(
                    "Persisted research job failed.",
                    extra={
                        "job_id": state["job_id"],
                        "event": "research_job_failed",
                    },
                )
                failure_message = "Research job failed unexpectedly."
                result = self._failed_result(state, failure_message)

            await self._finish_persisted_job(recorder, result, failure_message)
        finally:
            resources.clear()
            if self._llm_service is not None:
                self._llm_service.release_job(state["job_id"])

    async def _finish_persisted_job(
        self,
        recorder: PipelineRecorder,
        result: ResearchExecutionResult,
        failure_message: str | None,
    ) -> None:
        completed = result.status == ResearchExecutionStatus.COMPLETED
        details = {} if completed else _failure_details(result)
        if failure_message:
            details["reason"] = failure_message
        terminal_event = recorder.make(
            "workflow_completed" if completed else "workflow_failed",
            "workflow",
            "completed" if completed else "failed",
            details=details,
        )
        await self._job_store.finish_job(
            result=result,
            terminal_event=terminal_event,
            error_message=None if completed else details.get("reason"),
        )

    async def get_job(self, job_id: str) -> dict | None:
        return await self._job_store.get_job(job_id)

    async def get_events(self, job_id: str, *, after: int = 0):
        return await self._job_store.get_events(job_id, after=after)

    async def shutdown(self) -> None:
        """Stop local tasks before SQLite and shared providers are closed."""

        async with self._admission_lock:
            self._shutting_down = True
            tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    def _build_result(self, *, state: WorkflowState) -> ResearchExecutionResult:
        """Convert the final LangGraph state into an application result."""

        errors = list(state["errors"])
        final_report = state["writer"]["final_report"]
        terminal_errors = [error for error in errors if not error.recoverable]

        # A graph that ends without producing a report is not successful.
        if final_report is None and not terminal_errors:
            error = ResearchError(
                error_type=ErrorType.INTERNAL,
                component="research_service",
                message="Research workflow completed without producing a final report.",
                recoverable=False,
            )
            errors.append(error)
            terminal_errors.append(error)

        status = (
            ResearchExecutionStatus.FAILED if terminal_errors
            else ResearchExecutionStatus.COMPLETED
        )
        result = ResearchExecutionResult(
            job_id=state["job_id"],
            research_session_id=state["research_session_id"],
            status=status,
            stop_reason=state["research"]["stop_reason"],
            report=None if terminal_errors else final_report,
            errors=errors,
        )
        log_details = _failure_details(result)
        log = logger.error if terminal_errors else logger.info
        log(
            (
                "Research workflow execution failed."
                if terminal_errors
                else "Research workflow execution finished."
            ),
            extra={
                "event": (
                    "research_workflow_failed"
                    if terminal_errors
                    else "research_workflow_finished"
                ),
                "job_id": state["job_id"],
                "research_session_id": state["research_session_id"],
                "node": "research_service",
                "iteration": state["research"]["iteration_count"],
                "status": status.value,
                "error_type": log_details.get("error_type"),
                "component": log_details.get("component"),
                "data": {
                    "error_count": log_details.get("error_count", 0),
                    "reason": log_details.get("reason"),
                },
            },
        )
        return result
