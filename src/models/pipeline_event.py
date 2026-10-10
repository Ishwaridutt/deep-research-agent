"""Public API models for persisted research jobs and pipeline events."""

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.models import ResearchExecutionResult


JobStatus = Literal["queued", "running", "completed", "failed"]
EventType = Literal[
    "workflow_started", "stage_started", "stage_completed", "stage_failed",
    "research_retry", "research_stopped", "workflow_completed", "workflow_failed",
]


class PipelineEvent(BaseModel):
    """One sanitized, replayable workflow event."""

    job_id: str
    sequence: int = Field(ge=1)
    event_type: EventType
    stage: str
    status: str
    iteration: int | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResearchJobCreated(BaseModel):
    """Response returned when an asynchronous job is queued."""

    job_id: str
    research_session_id: str
    status: Literal["queued"] = "queued"


class ResearchJobStatusResponse(BaseModel):
    """Persisted job status, with a result only after completion."""

    job_id: str
    research_session_id: str
    question: str
    allowed_sources: list[str]
    status: JobStatus
    result: ResearchExecutionResult | None = None
    error_message: str | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None


    