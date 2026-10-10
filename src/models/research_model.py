"""Core research request, planning, and query domain models."""

from enum import Enum

from pydantic import BaseModel, Field


class ResearchSource(str, Enum):
    """Sources that a research job may be allowed to use."""

    WEB = "web"

    # Future phases.
    DOCUMENTS = "documents"
    MCP = "mcp"


class ResearchStopReason(str, Enum):
    """Explains why the research loop stopped."""

    COVERAGE_SUFFICIENT = "coverage_sufficient"
    MAX_ITERATIONS = "max_iterations"
    QUERY_BUDGET_EXHAUSTED = "query_budget_exhausted"
    EXTRACTION_BUDGET_EXHAUSTED = "extraction_budget_exhausted"
    TIME_BUDGET_EXHAUSTED = "time_budget_exhausted"
    NO_EVIDENCE_IMPROVEMENT = "no_evidence_improvement"
    NO_ACTIONABLE_GAPS = "no_actionable_gaps"


class ResearchExecutionStatus(str, Enum):
    """Overall outcome of one research workflow execution."""

    COMPLETED = "completed"
    FAILED = "failed"


class ResearchRequest(BaseModel):
    """Request accepted by the research workflow."""

    question: str = Field(
        min_length=3,
    )

    sources: list[ResearchSource] = Field(
        default_factory=lambda: [ResearchSource.WEB],
        min_length=1,
    )


class PlannedSubQuestion(BaseModel):
    """LLM-generated subquestion before application-owned IDs are assigned."""

    question: str = Field(
        min_length=1,
    )

    objective: str = Field(
        min_length=1,
    )


class PlannerOutput(BaseModel):
    """Structured output returned by the research planner."""

    subquestions: list[PlannedSubQuestion]


class SubQuestion(BaseModel):
    """A persisted subquestion belonging to the research plan."""

    id: str = Field(
        min_length=1,
    )

    question: str = Field(
        min_length=1,
    )

    objective: str = Field(
        min_length=1,
    )


class GeneratedSearchQuery(BaseModel):
    """LLM-generated query before system metadata is attached."""

    query: str = Field(
        min_length=1,
    )

    purpose: str = Field(
        min_length=1,
    )


class QueryGeneratorOutput(BaseModel):
    """Structured output returned by the query generator."""

    queries: list[GeneratedSearchQuery]


class SearchQuery(BaseModel):
    """A validated search query associated with one subquestion."""

    id: str = Field(
        min_length=1,
    )

    subquestion_id: str = Field(
        min_length=1,
    )

    query: str = Field(
        min_length=1,
    )

    purpose: str = Field(
        min_length=1,
    )

    iteration: int = Field(
        ge=1,
    )

