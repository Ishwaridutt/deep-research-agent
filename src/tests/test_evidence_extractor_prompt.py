"""Focused tests for the score-only prompt and chunk response schema."""

import json

import pytest
from pydantic import ValidationError

from src.models import EvidenceCandidate, EvidenceExtractionOutput, RetrievalChunk, SubQuestion
from src.prompts.evidence_extractor_prompt import (
    build_evidence_extractor_messages,
)


def test_evidence_prompt_scores_original_chunks_against_objective_and_queries():
    messages = build_evidence_extractor_messages(
        original_question="What affects muscle growth?",
        subquestion=SubQuestion(
            id="sq-1",
            question="How does protein affect hypertrophy?",
            objective="Compare findings, moderators, and limitations.",
        ),
        search_queries=["protein dose hypertrophy", "protein timing trial"],
        chunks=[RetrievalChunk(
            chunk_id="chunk-1", document_id="source-1", content="Study text.\nNo effect found.",
            source_type="web", chunk_index=0, token_count=8, title="Protein study",
        )],
    )

    system_prompt = messages[0].content
    user_prompt = messages[1].content
    normalized_prompt = " ".join(system_prompt.split())

    assert "exactly one chunk_id and relevance_score" in normalized_prompt
    assert "Do not extract, summarize, rewrite, or return source text" in normalized_prompt
    assert "need not answer every query" in normalized_prompt
    assert "contradictions, null results, and limitations" in normalized_prompt
    assert "untrusted data" in normalized_prompt
    context = json.loads(user_prompt.split("<scoring_context>\n")[1].split("\n</scoring_context>")[0])
    assert context["search_queries"] == ["protein dose hypertrophy", "protein timing trial"]
    assert context["objective"] == "Compare findings, moderators, and limitations."
    assert context["chunks"][0]["content"] == "Study text.\nNo effect found."
    assert context["chunks"][0]["chunk_id"] == "chunk-1"


def test_score_schema_does_not_ask_model_to_generate_content():
    schema = EvidenceCandidate.model_json_schema()
    assert set(schema["properties"]) == {"chunk_id", "relevance_score"}
    assert set(schema["required"]) == {"chunk_id", "relevance_score"}
    assert EvidenceExtractionOutput.model_validate({"evidence": [
        {"chunk_id": "chunk-1", "relevance_score": 0.8},
    ]}).evidence[0].relevance_score == 0.8
    with pytest.raises(ValidationError):
        EvidenceCandidate(chunk_id="chunk-1", relevance_score=0.8, content="Rewritten text")


@pytest.mark.parametrize("score", [-0.1, 1.1, float("nan"), float("inf")])
def test_score_schema_rejects_out_of_range_values(score):
    with pytest.raises(ValidationError):
        EvidenceCandidate(chunk_id="chunk-1", relevance_score=score)

