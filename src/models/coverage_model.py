"""Models used to evaluate research coverage and evidence quality."""

from enum import Enum

from pydantic import BaseModel, Field


class CoverageStatus(str, Enum):
    """Readiness to answer the core request, allowing disclosed minor gaps."""

    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"


class SubquestionCoverageStatus(str, Enum):
    """Coverage of an individual research objective."""

    SUPPORTED = "supported"
    PARTIAL = "partial"
    MISSING = "missing"
    CONFLICTING = "conflicting"


class EvidenceStrength(str, Enum):
    """Assessed strength of evidence supporting an objective."""

    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    UNKNOWN = "unknown"


class CoverageGap(BaseModel):
    """Describes missing or insufficient research evidence."""

    subquestion_id: str = Field(min_length=1)
    description: str = Field(min_length=1)

    priority: int = Field(default=3, ge=1, le=5)

    suggested_queries: list[str] = Field(
        default_factory=list,
    )


class SubquestionCoverage(BaseModel):
    """Quality and coverage assessment for one subquestion."""

    subquestion_id: str = Field(min_length=1)

    status: SubquestionCoverageStatus

    evidence_strength: EvidenceStrength = EvidenceStrength.UNKNOWN

    evidence_ids: list[str] = Field(
        default_factory=list,
    )

    reason: str = Field(min_length=1)


class CoverageResult(BaseModel):
    """Report readiness plus per-objective diagnostics and remaining gaps."""

    status: CoverageStatus

    gaps: list[CoverageGap] = Field(
        default_factory=list,
        description="Material blockers when insufficient; remaining report limitations when sufficient.",
    )

    assessments: list[SubquestionCoverage] = Field(
        default_factory=list,
    )
