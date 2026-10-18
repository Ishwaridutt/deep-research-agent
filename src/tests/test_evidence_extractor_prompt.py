"""Focused tests for comprehensive evidence-extraction instructions."""

from src.models import RetrievalResult, SubQuestion
from src.prompts.evidence_extractor_prompt import (
    build_evidence_extractor_messages,
)


def test_evidence_prompt_requests_high_recall_with_study_qualifiers():
    messages = build_evidence_extractor_messages(
        original_question="What affects muscle growth?",
        subquestion=SubQuestion(
            id="sq-1",
            question="How does protein affect hypertrophy?",
            objective="Compare findings, moderators, and limitations.",
        ),
        retrieval_result=RetrievalResult(
            subquestion_id="sq-1",
            retrieval_query="protein hypertrophy",
            chunks=[],
            context="[Evidence 1]\nChunk ID: chunk-1\nContent:\nStudy text.",
            estimated_tokens=10,
        ),
    )

    system_prompt = messages[0].content
    user_prompt = messages[1].content
    normalized_prompt = " ".join(system_prompt.split())

    assert "Optimize for high recall" in normalized_prompt
    assert "Examine every chunk" in normalized_prompt
    assert "sample size" in normalized_prompt
    assert "null findings" in normalized_prompt
    assert "Multiple items may use the same Chunk ID" in normalized_prompt
    assert "corroborating or conflicting claims" in normalized_prompt
    assert "Perform a high-recall extraction" in user_prompt

