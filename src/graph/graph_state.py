"""LangGraph state definitions for the research workflow."""

from typing import TypedDict

from src.models import (
    CitationCheckResult,
    CoverageResult,
    DraftReport,
    Evidence,
    ResearchError,
    ResearchReport,
    ResearchSource,
    ResearchStopReason,
    SearchQuery,
    SearchResult,
    Source,
    SubQuestion,
)

from src.models.research_quality import (
    ResearchBudget,
    ResearchControlState,
)

from src.models.retrieval_model import RetrievalResult


class ResearchAgentState(TypedDict):
    """
    Stores state owned by the Research Agent.

    The Research Agent is responsible for planning, search query
    generation, retrieval, evidence extraction, and research
    coverage evaluation.
    """

    subquestions: list[SubQuestion]
    search_queries: list[SearchQuery]
    search_results: list[SearchResult]
    sources: list[Source]
    retrieval_results: list[RetrievalResult]
    evidence: list[Evidence]

    iteration_count: int

    coverage: CoverageResult | None
    stop_reason: ResearchStopReason | None


class WriterAgentState(TypedDict):
    """
    Stores state owned by the Writer Agent.

    The Writer Agent receives selected research evidence and
    produces a grounded report that is later validated by
    the citation checker.
    """

    evidence_ids: list[str]

    draft_report: DraftReport | None
    citation_result: CitationCheckResult | None
    final_report: ResearchReport | None


class WorkflowState(TypedDict):
    """
    Represents the complete state of one research workflow.

    State ownership:

    research:
        Research planning, retrieval, evidence extraction,
        and coverage evaluation.

    research_control:
        Research budgets, usage, unresolved research gaps,
        and execution history.

    writer:
        Evidence preparation, report synthesis,
        and citation validation.
    """

    job_id: str
    research_session_id: str
    question: str

    allowed_sources: list[ResearchSource]

    research: ResearchAgentState

    research_control: ResearchControlState

    writer: WriterAgentState

    errors: list[ResearchError]


def create_initial_workflow_state(
    *,
    job_id: str,
    research_session_id: str,
    question: str,
    allowed_sources: list[ResearchSource],
    research_budget: ResearchBudget | None = None,
) -> WorkflowState:
    """
    Create the initial state for a research workflow.

    Centralized initialization ensures that every workflow
    receives complete research, control, and writer states.
    """

    return WorkflowState(
        job_id=job_id,
        research_session_id=research_session_id,
        question=question,
        allowed_sources=allowed_sources,
        research=ResearchAgentState(
            subquestions=[],
            search_queries=[],
            search_results=[],
            sources=[],
            retrieval_results=[],
            evidence=[],
            iteration_count=0,
            coverage=None,
            stop_reason=None,
        ),
        research_control=ResearchControlState(
            budget=research_budget or ResearchBudget(),
        ),
        writer=WriterAgentState(
            evidence_ids=[],
            draft_report=None,
            citation_result=None,
            final_report=None,
        ),
        errors=[],
    )

