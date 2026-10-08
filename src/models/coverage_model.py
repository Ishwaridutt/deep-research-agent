"""Models used to evaluate research coverage."""

from enum import Enum

from pydantic import BaseModel, Field


class CoverageStatus(str, Enum):
    """Overall evidence coverage state."""

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


class CoverageGap(BaseModel):
    """Describes evidence missing for a particular subquestion."""

    subquestion_id: str = Field(
        min_length=1,
    )

    description: str = Field(
        min_length=1,
    )


class CoverageResult(BaseModel):
    """Structured result returned by the coverage evaluator."""

    status: CoverageStatus

    gaps: list[CoverageGap] = Field(
        default_factory=list,
    )

    