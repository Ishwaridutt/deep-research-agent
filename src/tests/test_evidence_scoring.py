"""Offline relevance scoring, response validation, and evidence lineage tests."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from src.agents.research.research_evidence_extractor_node import research_evidence_extractor_node
from src.graph.graph_state import create_initial_workflow_state
from src.llm import LLMRole, LLMServiceError
from src.models import (
    Evidence, EvidenceExtractionOutput, ResearchSource, RetrievalChunk,
    RetrievalResult, RetrievedChunk, SearchQuery, SourceType, SubQuestion,
)


class FakeScorer:
    def __init__(self, *outputs):
        self.outputs = iter(outputs)
        self.calls = []

    async def invoke_structured(self, **kwargs):
        self.calls.append(kwargs)
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return output


def chunk(identifier, content=None):
    return RetrievalChunk(
        chunk_id=identifier, document_id=f"source-{identifier}",
        content=content if content is not None else f"Original passage {identifier}.",
        source_type="web", chunk_index=2, token_count=10, title="Source title",
        section="Results", metadata={"url": "https://example.com/study", "page_start": 3},
    )


def retrieval(chunks, subquestion_id="SQ-1"):
    return RetrievalResult(
        subquestion_id=subquestion_id, retrieval_query="metadata, not search lineage",
        chunks=[RetrievedChunk(item, index + 1) for index, item in enumerate(chunks)],
        context="Legacy formatted context is not the scoring input.", estimated_tokens=20,
    )


def state_with_chunks(*chunks):
    state = create_initial_workflow_state(
        job_id="job-1", research_session_id="session-1", question="Explain the findings",
        allowed_sources=[ResearchSource.WEB],
    )
    state["research"]["subquestions"] = [SubQuestion(
        id="SQ-1", question="What do the results show?", objective="Assess outcomes and uncertainty.",
    )]
    state["research"]["web_retrieval_results"] = [retrieval(chunks)]
    return state


def scores(*items):
    return EvidenceExtractionOutput(evidence=[
        {"chunk_id": identifier, "relevance_score": score} for identifier, score in items
    ])


def score_state(state, service, threshold=0.6, capacity=100):
    return asyncio.run(research_evidence_extractor_node(
        state, llm_service=service,
        settings=SimpleNamespace(research_limits=SimpleNamespace(
            max_evidence_items=capacity, min_evidence_relevance=threshold,
        )),
    ))


def test_scoring_audit_records_real_decisions_and_query_matches_without_private_text():
    public = chunk("public", "Public finding " * 100)
    private = chunk("private", "CONFIDENTIAL")
    private.source_type = "document"
    private.title = "secret-file.pdf"
    private.metadata = {"private_document_id": "secret-document", "owner_id": "secret-owner"}
    state = state_with_chunks(public, private, chunk("missing"), chunk("low"))
    state["research"]["web_retrieval_results"][0].query_matches = [{
        "query": "specific search", "selected_chunk_ids": ["public"],
    }]
    result = score_state(state, FakeScorer(scores(("public", .9), ("private", .7), ("low", .1))))
    rows = result["research"]["evidence_audit"][0]["passages"]
    assert [row["decision"] for row in rows] == ["accepted", "accepted", "missing_score", "below_threshold"]
    assert rows[0]["relevance_score"] == .9 and rows[0]["queries"] == ["specific search"]
    assert len(rows[0]["preview"]) == 800 and rows[0]["preview_truncated"]
    audit = json.dumps(rows)
    for secret in ("CONFIDENTIAL", "secret-file.pdf", "secret-document", "secret-owner"):
        assert secret not in audit


@pytest.mark.parametrize("threshold,expected", [(0, ["high", "middle", "low"]), (0.6, ["high", "middle"]), (1, ["high"])])
def test_configured_threshold_filters_original_chunks_including_boundary(threshold, expected, caplog):
    original = chunk("middle", "  Trial: n=50; no clear effect.\n95% CI −1 to 2.  ")
    state = state_with_chunks(chunk("low"), original, chunk("high"))
    service = FakeScorer(scores(("low", 0.1), ("middle", 0.6), ("high", 1)))

    with caplog.at_level("INFO"):
        result = score_state(state, service, threshold=threshold)

    evidence = result["research"]["evidence"]
    assert [item.chunk_id for item in evidence] == expected
    if "middle" in expected:
        accepted = next(item for item in evidence if item.chunk_id == "middle")
        assert accepted.content == original.content
        assert accepted.provenance["page_start"] == 3
        assert accepted.provenance["section"] == "Results"
        assert accepted.provenance["chunk_index"] == 2
    assert not result["errors"]
    assert service.calls[0]["role"] == LLMRole.EVIDENCE_EXTRACTOR
    assert service.calls[0]["output_schema"] is EvidenceExtractionOutput
    stats = result["research"]["evidence_stats"]
    assert stats["scored_chunk_count"] == 3
    assert stats["accepted_chunk_count"] == len(expected)
    assert stats["low_relevance_chunk_count"] == 3 - len(expected)
    assert stats["total_evidence_count"] == len(expected)
    record = next(item for item in caplog.records if getattr(item, "event", "") == "evidence_extraction_completed")
    assert record.data == {
        "scored_chunk_count": 3, "low_relevance_chunk_count": 3 - len(expected),
        "accepted_chunk_count": len(expected), "min_evidence_relevance": threshold,
    }


def test_only_current_task_queries_are_supplied_and_followup_keeps_original_objective():
    state = state_with_chunks(chunk("new"))
    state["research"]["iteration_count"] = 1
    state["research"]["active_subquestions"] = [SubQuestion(
        id="SQ-followup", parent_subquestion_id="SQ-1",
        question="What explains the uncertainty?", objective="Find the missing measurement.",
    )]
    state["research"]["web_retrieval_results"][0].subquestion_id = "SQ-followup"
    state["research"]["search_queries"] = [SearchQuery(
        id=identifier, subquestion_id=task, query=text, purpose="Find data", iteration=iteration,
    ) for identifier, task, text, iteration in [
        ("query-old", "SQ-followup", "old query", 1),
        ("query-other", "SQ-other", "other task", 2),
        ("query-1", "SQ-followup", "measurement trial", 2),
        ("query-2", "SQ-followup", "measurement uncertainty", 2),
    ]]
    service = FakeScorer(scores(("new", 0.9)))

    result = score_state(state, service)

    call = service.calls[0]
    context = json.loads(call["messages"][1].content.split("<scoring_context>\n")[1].split("\n</scoring_context>")[0])
    assert context["search_queries"] == ["measurement trial", "measurement uncertainty"]
    assert context["subquestion"] == "What explains the uncertainty?"
    accepted = result["research"]["evidence"][0]
    assert accepted.subquestion_id == "SQ-1"
    assert accepted.provenance["research_subquestion_id"] == "SQ-followup"
    assert accepted.provenance["research_iteration"] == 2
    assert accepted.provenance["research_query_ids"] == ["query-1", "query-2"]


def test_unknown_duplicate_and_missing_scores_are_rejected_and_logged(caplog):
    state = state_with_chunks(chunk("valid"), chunk("duplicate"), chunk("missing"))
    service = FakeScorer(scores(
        ("unknown", 1), ("valid", 0.8), ("duplicate", 0.9), ("duplicate", 0.1),
    ))

    result = score_state(state, service)

    assert [item.chunk_id for item in result["research"]["evidence"]] == ["valid"]
    assert len(result["errors"]) == 1
    assert result["errors"][0].recoverable
    assert result["research"]["evidence_stats"]["missing_chunk_count"] == 1
    assert result["research"]["evidence_stats"]["unknown_chunk_count"] == 1
    assert result["research"]["evidence_stats"]["duplicate_chunk_count"] == 1
    record = next(item for item in caplog.records if getattr(item, "event", "") == "evidence_scores_invalid")
    assert record.data == {"missing_chunk_count": 1, "unknown_chunk_count": 1, "duplicate_chunk_count": 1}


def test_old_evidence_is_preserved_and_not_rescored_for_the_same_objective():
    state = state_with_chunks(chunk("old"), chunk("new"))
    old = Evidence(
        id="EV-old", subquestion_id="SQ-1", source_id="source-old", chunk_id="old",
        source_type=SourceType.WEB, content="Previously accepted passage.", relevance_score=0.7,
    )
    state["research"]["evidence"] = [old]
    service = FakeScorer(scores(("new", 0.9)))

    result = score_state(state, service)

    assert result["research"]["evidence"][0] is old
    assert [item.chunk_id for item in result["research"]["evidence"]] == ["old", "new"]
    context = service.calls[0]["messages"][1].content
    assert '"chunk_id": "old"' not in context
    state["research"] = result["research"]
    assert score_state(state, FakeScorer())["research"]["evidence"] == result["research"]["evidence"]


def test_same_chunk_can_support_distinct_original_objectives():
    state = state_with_chunks(chunk("shared"))
    state["research"]["subquestions"].append(SubQuestion(
        id="SQ-2", question="What are the limitations?", objective="Understand uncertainty.",
    ))
    state["research"]["web_retrieval_results"].append(retrieval([chunk("shared")], "SQ-2"))

    result = score_state(state, FakeScorer(scores(("shared", 0.9)), scores(("shared", 0.8))))

    assert [item.subquestion_id for item in result["research"]["evidence"]] == ["SQ-1", "SQ-2"]


def test_capacity_admits_strongest_scores_and_skips_call_when_full():
    state = state_with_chunks(chunk("first"), chunk("strongest"))
    result = score_state(state, FakeScorer(scores(("first", 0.7), ("strongest", 0.95))), capacity=1)
    assert [item.chunk_id for item in result["research"]["evidence"]] == ["strongest"]
    assert result["research"]["evidence_stats"]["capacity_skipped_chunk_count"] == 1
    state["research"] = result["research"]
    service = FakeScorer()
    result = score_state(state, service, capacity=1)
    assert result["research"]["evidence"] == state["research"]["evidence"]
    assert result["research"]["evidence_stats"]["capacity_reached"]
    assert result["research"]["evidence_stats"]["scored_chunk_count"] == 0
    assert result["research"]["evidence_stats"]["accepted_chunk_count"] == 0
    assert not service.calls


def test_provider_failure_is_recoverable_and_other_source_results_continue():
    state = state_with_chunks(chunk("failed"))
    state["research"]["document_retrieval_results"] = [retrieval([chunk("accepted")])]
    state["allowed_sources"].append(ResearchSource.DOCUMENTS)
    service = FakeScorer(
        LLMServiceError(role=LLMRole.EVIDENCE_EXTRACTOR, message="Provider unavailable"),
        scores(("accepted", 0.9)),
    )

    result = score_state(state, service)

    assert [item.chunk_id for item in result["research"]["evidence"]] == ["accepted"]
    assert result["errors"][0].recoverable
    assert result["research"]["evidence_stats"]["failed_batch_count"] == 1
    assert all(call["metadata"].contains_private_data for call in service.calls)


def test_blank_or_unknown_task_chunks_skip_calls_and_empty_scores_do_not_default_to_acceptance():
    state = state_with_chunks(chunk("empty", " \n "))
    state["research"]["web_retrieval_results"].append(retrieval([chunk("orphan")], "unknown"))
    service = FakeScorer()
    result = score_state(state, service)
    assert not result["research"]["evidence"]
    assert not service.calls

    result = score_state(state_with_chunks(chunk("unscored")), FakeScorer(scores()))
    assert not result["research"]["evidence"]
    assert result["errors"][0].recoverable
