"""Core research request, planning, and query domain models."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from src.models.document_models import ResearchSourceSelection


class ResearchSource(str, Enum):
    """Sources that a research job may be allowed to use."""

    WEB = "web"
    DOCUMENTS = "documents"

    # Future phases.
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

    source_selection: ResearchSourceSelection = Field(
        default_factory=ResearchSourceSelection,
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


class PlannedFollowupSubQuestion(PlannedSubQuestion):
    """A gap-focused task linked to an original research objective."""

    parent_subquestion_id: str = Field(min_length=1)


class FollowupPlannerOutput(BaseModel):
    """Refinement may return no tasks when another search would not help."""

    subquestions: list[PlannedFollowupSubQuestion]


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

    parent_subquestion_id: str | None = None
    iteration: int = Field(default=1, ge=1)


# Search branches a query may use. General web search always runs; the native
# arXiv and Wikipedia branches run only for queries that request them.
SearchProviderName = Literal["web", "arxiv", "wikipedia"]


class GeneratedSearchQuery(BaseModel):
    """LLM-generated query before system metadata is attached."""

    query: str = Field(
        min_length=1,
    )

    purpose: str = Field(
        min_length=1,
    )

    providers: list[SearchProviderName] = Field(
        default_factory=lambda: ["web"],
        description=(
            'Search branches for this query. Always include "web". Add "arxiv" only '
            'for computer science, maths, physics, statistics, or quantitative-biology '
            'research papers; add "wikipedia" only for definitions, background, entities, or history.'
        ),
    )


class SubquestionSearchQueries(BaseModel):
    """Queries generated for one subquestion within a batched generator call."""

    subquestion_id: str = Field(
        min_length=1,
    )

    queries: list[GeneratedSearchQuery]


class QueryGeneratorOutput(BaseModel):
    """Structured output returned by one batched query generator call."""

    subquestions: list[SubquestionSearchQueries]


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

    providers: list[SearchProviderName] = Field(
        default_factory=lambda: ["web"],
    )

