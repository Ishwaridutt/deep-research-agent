"""Application-level research execution result models."""

from pydantic import BaseModel, Field

from src.models.errors_model import ResearchError
from src.models.report_model import ResearchReport
from src.models.research_model import (
    ResearchExecutionStatus,
    ResearchStopReason,
)


class ResearchExecutionResult(BaseModel):
    """
    Final application-level result of one research workflow execution.

    This is returned by ResearchService and later mapped to an API response.
    """

    job_id: str = Field(
        min_length=1,
    )

    research_session_id: str = Field(
        min_length=1,
    )

    status: ResearchExecutionStatus
    stop_reason: ResearchStopReason | None = None
    report: ResearchReport | None = None
    errors: list[ResearchError] = Field(
        default_factory=list,
    )
    

