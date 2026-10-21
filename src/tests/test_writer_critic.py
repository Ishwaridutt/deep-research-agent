"""Offline tests for the critic review, its single revision, and sentence-level citation trimming."""

import asyncio
from datetime import UTC, datetime

from src.agents.writer.writer_citation_checker_node import _remove_sentences, writer_citation_checker_node
from src.agents.writer.writer_critic_node import writer_critic_node
from src.agents.writer.writer_synthesis_node import writer_synthesis_node
from src.graph.graph_routing import route_after_critic, route_after_synthesis
from src.graph.graph_state import create_initial_workflow_state
from src.llm import LLMRole, LLMServiceError
from src.models import (
    CriticIssue, CriticReview, DraftReport, Evidence, ReportFinding, ResearchSource,
    SemanticCitationCheckOutput, Source, SourceProvider, SourceType,
)


class FakeLLM:
    def __init__(self, *outputs):
        self.outputs = iter(outputs)
        self.calls = []

    async def invoke_structured(self, **kwargs):
        self.calls.append(kwargs)
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return output


def writer_state():
    state = create_initial_workflow_state(
        job_id="job-1", research_session_id="session-1",
        question="How much protein and fat should I eat?", allowed_sources=[ResearchSource.WEB],
    )
    for index in (1, 2):
        state["research"]["evidence"].append(Evidence(
            id=f"EV-{index}", subquestion_id="SQ-1", source_id=f"SRC-{index}",
            chunk_id=f"SRC-{index}:0", source_type=SourceType.WEB,
            content=f"Supported detail {index}.", relevance_score=0.9,
        ))
        state["research"]["sources"].append(Source(
            id=f"SRC-{index}", source_type=SourceType.WEB, provider=SourceProvider.TAVILY,
            canonical_id=f"url:https://example.com/{index}", title=f"Source {index}",
            uri=f"https://example.com/{index}", content="Body.", retrieved_at=datetime.now(UTC),
        ))
    state["writer"]["evidence_ids"] = ["EV-1", "EV-2"]
    state["writer"]["draft_report"] = DraftReport(
        executive_summary="Protein matters. [EV-1]",
        findings=[ReportFinding(title="Protein", content="Eat 1.6 g/kg of protein. [EV-1]", evidence_ids=["EV-1"])],
    )
    return state


REVISE = CriticReview(verdict="revise", issues=[CriticIssue(
    finding_index=None, issue_type="missing_requested_part", instruction="Add the fat intake range from EV-2.",
)])


def test_critic_revision_runs_once_with_feedback_then_goes_to_citation_check():
    state = writer_state()
    revised = DraftReport(
        executive_summary="Protein and fat ranges. [EV-1][EV-2]",
        findings=[ReportFinding(title="Macros", content="Protein and fat. [EV-1][EV-2]", evidence_ids=["EV-1", "EV-2"])],
    )
    llm = FakeLLM(REVISE, revised)

    async def run():
        assert route_after_synthesis(state) == "critic"
        state.update(await writer_critic_node(state, llm_service=llm))
        assert route_after_critic(state) == "revise"
        state.update(await writer_synthesis_node(state, llm_service=llm))
        # The revised draft skips the critic and goes straight to citation checking.
        assert route_after_synthesis(state) == "citation_check"
        assert route_after_critic(state) == "citation_check"

    asyncio.run(run())
    assert [call["role"] for call in llm.calls] == [LLMRole.CRITIC, LLMRole.WRITER]
    revision_prompt = llm.calls[1]["messages"][1].content
    assert "<previous_draft>" in revision_prompt and "Add the fat intake range" in revision_prompt
    assert state["writer"]["draft_report"] == revised
    assert state["writer"]["revision_count"] == 1


def test_critic_approval_and_failure_both_continue_to_citation_check():
    approved = writer_state()
    approved.update(asyncio.run(writer_critic_node(approved, llm_service=FakeLLM(CriticReview(verdict="approve")))))
    assert route_after_critic(approved) == "citation_check"

    failed = writer_state()
    failed.update(asyncio.run(writer_critic_node(failed, llm_service=FakeLLM(
        LLMServiceError(role=LLMRole.CRITIC, message="Provider unavailable"),
    ))))
    assert failed["writer"]["critic_review"].verdict == "approve"
    assert failed["errors"][0].recoverable and failed["errors"][0].component == "writer_critic"
    assert route_after_critic(failed) == "citation_check"

    # "revise" without any issue gives the writer nothing to fix.
    empty = writer_state()
    empty.update(asyncio.run(writer_critic_node(empty, llm_service=FakeLLM(CriticReview(verdict="revise")))))
    assert route_after_critic(empty) == "citation_check"


def test_critic_skips_llm_when_there_is_nothing_to_review():
    state = writer_state()
    state["writer"]["draft_report"] = DraftReport(executive_summary="No evidence.", findings=[])
    llm = FakeLLM()
    state.update(asyncio.run(writer_critic_node(state, llm_service=llm)))
    assert llm.calls == [] and state["writer"]["critic_review"].verdict == "approve"


def test_failed_revision_keeps_the_reviewed_draft():
    state = writer_state()
    reviewed = state["writer"]["draft_report"]
    state["writer"]["critic_review"] = REVISE
    state.update(asyncio.run(writer_synthesis_node(state, llm_service=FakeLLM(
        LLMServiceError(role=LLMRole.WRITER, message="Provider unavailable"),
    ))))
    assert state["writer"]["draft_report"] == reviewed
    assert state["errors"][0].recoverable
    assert route_after_synthesis(state) == "citation_check"


def test_sentence_removal_ignores_citations_and_keeps_decimals_intact():
    content = "Eat 1.6–2.0 g/kg of protein daily. [EV-1] Fat should be 20–35% of calories [EV-2]. Carbs fill the rest."
    remaining, removed = _remove_sentences(content, ["Fat should be 20–35% of calories."])
    assert remaining == "Eat 1.6–2.0 g/kg of protein daily. [EV-1] Carbs fill the rest."
    assert removed == ["Fat should be 20–35% of calories [EV-2]."]
    assert _remove_sentences(content, ["Not in the text at all, clearly."]) == (content, [])


def test_partly_supported_finding_keeps_supported_sentences():
    state = writer_state()
    state["writer"]["draft_report"] = DraftReport(
        executive_summary="Protein matters. [EV-1]",
        findings=[ReportFinding(
            title="Macros",
            content="Eat 1.6 g/kg of protein daily. [EV-1] Fat should always be exactly 50% of calories. [EV-2]",
            evidence_ids=["EV-1", "EV-2"],
        )],
    )
    llm = FakeLLM(SemanticCitationCheckOutput(
        revised_executive_summary="Protein matters. [EV-1]",
        finding_results=[{
            "finding_index": 0, "supported": False, "reason": "The fat claim exceeds EV-2.",
            "unsupported_sentences": ["Fat should always be exactly 50% of calories."],
        }],
    ))
    state.update(asyncio.run(writer_citation_checker_node(state, llm_service=llm)))
    report = state["writer"]["final_report"]

    assert [finding.content for finding in report.findings] == ["Eat 1.6 g/kg of protein daily. [1]"]
    assert [source.source_id for source in report.sources] == ["SRC-1"]
    assert report.unverified_material[0].content == "Fat should always be exactly 50% of calories."
    assert [source.source_id for source in report.unverified_material[0].sources] == ["SRC-2"]


def test_semantic_check_failure_keeps_reference_checked_findings_with_a_limitation():
    state = writer_state()
    llm = FakeLLM(LLMServiceError(role=LLMRole.CITATION_CHECKER, message="model_not_found"))
    state.update(asyncio.run(writer_citation_checker_node(state, llm_service=llm)))
    report = state["writer"]["final_report"]

    # The job does not fail; the reference-checked finding is kept and the gap is disclosed.
    assert [finding.title for finding in report.findings] == ["Protein"]
    assert report.executive_summary == "Protein matters. [1]"
    assert any("Semantic citation verification was unavailable" in item for item in report.limitations)
    assert state["errors"][0].recoverable and state["errors"][0].component == "writer_citation_checker"
