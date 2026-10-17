"""Focused tests for generic multi-round research quality controls."""

from src.agents.research.research_query_generator_node import _allocate_query_slots
from src.models import (
    CoverageResult,
    CoverageStatus,
    Evidence,
    SourceType,
    SubQuestion,
)
from src.prompts.coverage_evaluator_prompt import build_coverage_evaluator_messages
from src.prompts.planner_prompt import build_planner_messages
from src.prompts.query_generator_prompt import build_query_generator_messages


def _subquestion(identifier: str) -> SubQuestion:
    return SubQuestion(
        id=identifier,
        question=f"Question {identifier}?",
        objective=f"Collect evidence for {identifier}.",
    )


def test_query_slots_are_allocated_fairly_before_second_queries():
    subquestions = [_subquestion("sq-1"), _subquestion("sq-2"), _subquestion("sq-3")]

    allocations = _allocate_query_slots(
        subquestions,
        remaining_budget=4,
        max_per_subquestion=2,
    )

    assert allocations == {"sq-1": 2, "sq-2": 1, "sq-3": 1}


def test_research_prompts_ground_dates_and_followup_context():
    current_date = "2026-10-16"
    planner = build_planner_messages(
        question="Find recent developments from the last two years.",
        max_subquestions=5,
        current_date=current_date,
    )
    query = build_query_generator_messages(
        original_question="Find recent developments.",
        subquestion=_subquestion("sq-1"),
        max_queries=2,
        previous_queries=["previous query"],
        current_date=current_date,
        coverage_gap="A direct comparison is missing.",
        suggested_queries=["Entity A Entity B comparison 2025 2026"],
        previous_evidence=["Entity A reported measurable growth."],
    )

    assert current_date in planner[0].content
    assert "exact date or year range" in planner[0].content
    assert "final report" in planner[0].content
    assert "same selected set" in planner[0].content
    assert current_date in query[0].content
    assert "fixed scope" in query[0].content
    assert "primary and directly attributable" in query[0].content
    assert "Entity A Entity B comparison 2025 2026" in query[1].content
    assert "Entity A reported measurable growth" in query[1].content


def test_coverage_prompt_includes_safe_source_context():
    subquestion = _subquestion("sq-1")
    evidence = Evidence(
        id="evidence-1",
        subquestion_id=subquestion.id,
        source_id="source-1",
        chunk_id="source-1:0",
        source_type=SourceType.WEB,
        provenance={
            "title": "Annual report",
            "url": "https://example.com/report",
            "provider": "tavily",
            "provider_metadata": {"published_at": "2026-03-01"},
        },
        content="The report documents the measured outcome.",
        relevance_score=0.9,
    )

    messages = build_coverage_evaluator_messages(
        original_question="What changed?",
        subquestions=[subquestion],
        evidence_by_subquestion={subquestion.id: [evidence]},
        current_date="2026-10-16",
        previous_coverage=CoverageResult(
            status=CoverageStatus.SUFFICIENT,
            gaps=[],
            assessments=[],
        ),
    )

    assert "2026-10-16" in messages[0].content
    assert "https://example.com/report" in messages[1].content
    assert "2026-03-01" in messages[1].content
    assert "directly executable" in messages[0].content
    assert "previously supported" in messages[0].content
    assert '"previous_coverage":{"status":"sufficient"' in messages[1].content
    assert "exactly that same set" in messages[0].content
