"""Focused tests for generic multi-round research quality controls."""

import json

from src.models import (
    CoverageResult,
    CoverageStatus,
    Evidence,
    SourceType,
    SubQuestion,
)
from src.prompts.coverage_evaluator_prompt import build_coverage_evaluator_messages
from src.prompts.research_planner_prompt import build_planner_messages
from src.prompts.query_generator_prompt import build_query_generator_messages


def _subquestion(identifier: str) -> SubQuestion:
    return SubQuestion(
        id=identifier,
        question=f"Question {identifier}?",
        objective=f"Collect evidence for {identifier}.",
    )


def test_research_prompts_ground_dates_and_followup_context():
    current_date = "2026-10-16"
    planner = build_planner_messages(
        question="Find recent developments from the last two years.",
        max_subquestions=5,
        current_date=current_date,
    )
    query = build_query_generator_messages(
        original_question="Find recent developments.",
        subquestions=[_subquestion("sq-1")],
        max_queries=2,
        max_query_words=12,
        previous_queries=["previous query"],
        current_date=current_date,
    )

    assert current_date in planner[0].content
    assert "Resolve relative dates" in planner[0].content
    assert "consistent entities and criteria" in planner[0].content
    assert "no more than 5 subquestions" in planner[0].content
    assert "Follow-up planning:" not in planner[0].content
    assert current_date in query[0].content
    assert "scope and entities fixed" in query[0].content
    assert "Resolve relative dates" in query[0].content
    assert "previous query" in query[1].content
    normalized_query_prompt = " ".join(query[0].content.split()).casefold()
    assert "do not answer the question or generate subquestions" in normalized_query_prompt
    assert "no more than 2 queries per subquestion" in normalized_query_prompt
    assert "short purpose" in query[0].content


def test_planner_refinement_keeps_lineage_and_history_with_concise_rules():
    context = {
        "original_subquestions": [{"id": "sq-1", "question": "What changed?"}],
        "unresolved_gaps": [{"subquestion_id": "sq-1", "description": "Missing measurements."}],
        "previous_tasks": [{"id": "sq-1", "queries": ["earlier search"]}],
        "available_query_slots": 3,
        "max_queries_per_subquestion": 2,
    }
    messages = build_planner_messages(
        question="Explain the changes.", max_subquestions=3,
        current_date="2026-10-18", refinement_context=context,
    )

    assert "no more than 3 subquestions" in messages[0].content
    assert "parent_subquestion_id from an unresolved gap" in messages[0].content
    assert "original objectives fixed" in messages[0].content
    assert "empty subquestions list" in messages[0].content
    payload = messages[1].content.split("<refinement_context>\n")[1].split("\n</refinement_context>")[0]
    assert json.loads(payload) == context


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
    assert "previously supported" in messages[0].content
    assert '"previous_coverage":{"status":"sufficient"' in messages[1].content
    assert "blocking=true only when a part the user explicitly asked" in " ".join(messages[0].content.split())
    assert "a generic passage" in messages[0].content
    assert "Use only names found in the evidence" in messages[0].content
