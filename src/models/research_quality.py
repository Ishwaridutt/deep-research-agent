"""Models for research quality and execution control."""

from datetime import datetime, timezone
from typing import Literal
from pydantic import BaseModel, Field

from src.models.coverage_model import CoverageGap

# defines execution limits.
class ResearchBudget(BaseModel):
    """Execution limits for one research workflow."""

    max_rounds: int = Field(
        default=3,
        ge=1,
    )

    max_search_queries: int = Field(
        default=30,
        ge=1,
    )

    max_extracted_pages: int = Field(
        default=60,
        ge=1,
    )

    max_extracted_pages_per_round: int = Field(
        default=20,
        ge=1,
    )

    max_research_seconds: float = Field(
        default=600.0,
        gt=0,
    )

    max_consecutive_no_gain_rounds: int = Field(
        default=1,
        ge=1,
    )

# tracks resource consumption. 
# The existing iteration_count remains the source of truth for the number of research rounds.
class ResearchUsage(BaseModel):
    """Resources consumed by the research workflow."""

    queries_executed: int = Field(
        default=0,
        ge=0,
    )

    page_extraction_attempts: int = Field(
        default=0,
        ge=0,
    )

    consecutive_no_gain_rounds: int = Field(
        default=0,
        ge=0,
    )

    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )

# records why research continued or stopped. 
# Its action values match your existing graph routing destinations.
class ResearchDecision(BaseModel):
    """Decision produced by the research routing logic."""

    action: Literal[
        "continue_research",
        "write_report",
        "terminal_error",
    ]

    reason: str

    has_unresolved_gaps: bool = False

    budget_exhausted: bool = False

# preserves the outcome of previous research iterations.
class ResearchRoundRecord(BaseModel):
    """Summary of one completed research iteration."""

    iteration: int = Field(
        ge=1,
    )

    new_evidence_ids: list[str] = Field(
        default_factory=list,
    )

    unresolved_gaps: list[CoverageGap] = Field(
        default_factory=list,
    )

    decision: ResearchDecision

# combines these into an independently managed state object.
class ResearchControlState(BaseModel):
    """
    Execution and quality-control state for the research workflow.

    ResearchAgentState continues to own evidence and coverage.
    This model owns research budgets, retry targets, and
    execution history.
    """

    budget: ResearchBudget = Field(
        default_factory=ResearchBudget,
    )

    usage: ResearchUsage = Field(
        default_factory=ResearchUsage,
    )

    pending_gaps: list[CoverageGap] = Field(
        default_factory=list,
    )

    history: list[ResearchRoundRecord] = Field(
        default_factory=list,
    )

    last_decision: ResearchDecision | None = None

