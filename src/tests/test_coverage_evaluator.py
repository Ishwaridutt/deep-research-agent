"""Offline coverage readiness, reference integrity, and early-stop regressions."""

import asyncio
from types import SimpleNamespace

import pytest

from src.agents.research.research_coverage_evaluator_node import (
    _group_evidence_by_subquestion,
    _normalize_coverage_result,
    research_coverage_evaluator_node,
)
from src.agents.writer.writer_synthesis_node import writer_synthesis_node
from src.graph.graph_routing import research_routing_node, route_after_research
from src.graph.graph_state import create_initial_workflow_state
from src.llm import LLMRole, LLMServiceError
from src.models import (
    CoverageGap, CoverageResult, CoverageStatus, DraftReport, Evidence,
    ResearchSource, ResearchStopReason, SourceType, SubQuestion,
)
from src.models.coverage_model import EvidenceStrength, SubquestionCoverage, SubquestionCoverageStatus


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


def initial_state():
    state = create_initial_workflow_state(
        job_id="job-1", research_session_id="session-1",
        question="Explain how the system works", allowed_sources=[ResearchSource.WEB],
    )
    state["research"]["subquestions"] = [SubQuestion(
        id="SQ-1", question="How does it work?", objective="Explain the mechanism.",
    )]
    state["research"]["evidence"] = [Evidence(
        id="EV-1", subquestion_id="SQ-1", source_id="SRC-1", chunk_id="SRC-1:0",
        source_type=SourceType.WEB, content="A specific explanation with documented details.",
        relevance_score=0.9,
    )]
    return state


def assessment(identifier="SQ-1", status="supported", ids=None, strength="moderate"):
    return SubquestionCoverage(
        subquestion_id=identifier, status=status, evidence_ids=ids if ids is not None else ["EV-1"],
        evidence_strength=strength, reason="The passage answers the question with stated qualifications.",
    )


def normalize(state, coverage):
    research = state["research"]
    return _normalize_coverage_result(
        coverage=coverage, subquestions=research["subquestions"],
        evidence_by_subquestion=_group_evidence_by_subquestion(
            subquestions=research["subquestions"], evidence=research["evidence"],
        ),
    )


def route(state):
    state.update(research_routing_node(state, settings=SimpleNamespace(
        research_limits=SimpleNamespace(max_iterations=3, max_total_queries_per_job=30),
    )))
    return route_after_research(state)


@pytest.mark.parametrize("strength", list(EvidenceStrength))
def test_one_specific_passage_can_be_sufficient_without_a_strength_quota(strength):
    state = initial_state()
    result = normalize(state, CoverageResult(
        status="sufficient", assessments=[assessment(strength=strength)],
    ))

    assert result.status == CoverageStatus.SUFFICIENT
    assert result.assessments[0].status == SubquestionCoverageStatus.SUPPORTED
    assert result.assessments[0].evidence_strength == strength
    assert not result.gaps


@pytest.mark.parametrize("status", ["supported", "partial", "conflicting"])
def test_qualified_report_can_be_ready_without_dropping_minor_gaps(status):
    state = initial_state()
    gap = CoverageGap(subquestion_id="SQ-1", description="Long-term measurements are not provided.")
    result = normalize(state, CoverageResult(
        status="sufficient", assessments=[assessment(status=status)], gaps=[gap],
    ))

    assert result.status == CoverageStatus.SUFFICIENT
    assert result.gaps == [gap]
    assert result.assessments[0].status.value == status


def test_missing_peripheral_objective_does_not_automatically_block_a_grounded_core_answer():
    state = initial_state()
    state["research"]["subquestions"].append(SubQuestion(
        id="SQ-2", question="What additional historical examples exist?", objective="Optional background.",
    ))
    result = normalize(state, CoverageResult(
        status="sufficient", assessments=[assessment(), assessment("SQ-2", "missing", [])],
        gaps=[CoverageGap(subquestion_id="SQ-2", description="Additional historical examples were not found.")],
    ))

    assert result.status == CoverageStatus.SUFFICIENT
    assert result.assessments[1].status == SubquestionCoverageStatus.MISSING
    assert len(result.gaps) == 1


def test_material_gap_preserves_the_planner_retry_route():
    state = initial_state()
    gap = CoverageGap(subquestion_id="SQ-1", description="The central mechanism remains unexplained.", blocking=True)
    limitation = CoverageGap(subquestion_id="SQ-1", description="Long-term data is limited.")
    state["research"]["coverage"] = normalize(state, CoverageResult(
        status="insufficient", assessments=[assessment(status="partial")], gaps=[gap, limitation],
    ))
    state["research"]["iteration_count"] = 1

    assert route(state) == "continue_research"
    # Only blocking gaps drive follow-up planning.
    assert state["research_control"].pending_gaps == [gap]
    assert state["research"]["stop_reason"] is None


def test_non_blocking_gaps_stop_research_even_when_the_model_says_insufficient():
    state = initial_state()
    limitation = CoverageGap(subquestion_id="SQ-1", description="Report limitation: optimal value is uncertain.", priority=3)
    state["research"]["coverage"] = normalize(state, CoverageResult(
        status="insufficient", assessments=[assessment(status="partial")], gaps=[limitation],
    ))
    state["research"]["iteration_count"] = 1

    assert state["research"]["coverage"].status == CoverageStatus.SUFFICIENT
    assert state["research"]["coverage"].gaps == [limitation]
    assert route(state) == "write_report"
    assert state["research"]["stop_reason"] == ResearchStopReason.COVERAGE_SUFFICIENT


def test_requested_part_without_evidence_or_judgment_becomes_a_blocking_gap():
    state = initial_state()
    state["research"]["subquestions"].append(SubQuestion(
        id="SQ-2", question="What fat intake is recommended?", objective="Find fat intake ranges.",
    ))
    result = normalize(state, CoverageResult(
        status="sufficient", assessments=[assessment(), assessment("SQ-2", "missing", [])],
    ))
    assert result.status == CoverageStatus.INSUFFICIENT
    assert [(gap.subquestion_id, gap.blocking) for gap in result.gaps] == [("SQ-2", True)]


def test_minor_gaps_stop_after_one_round_and_reach_the_writer_without_spending_the_budget():
    state = initial_state()
    state["allowed_sources"].append(ResearchSource.DOCUMENTS)
    state["writer"]["evidence_ids"] = ["EV-1"]
    service = FakeLLM(
        CoverageResult(
            status="sufficient", assessments=[assessment(status="partial")],
            gaps=[CoverageGap(subquestion_id="SQ-1", description="Minor long-term data limitation.")],
        ),
        DraftReport(executive_summary="Supported explanation. [EV-1]"),
    )

    async def run():
        state.update(await research_coverage_evaluator_node(state, llm_service=service))
        assert route(state) == "write_report"
        state.update(await writer_synthesis_node(state, llm_service=service))

    asyncio.run(run())
    assert state["research"]["iteration_count"] == 1
    assert state["research"]["stop_reason"] == ResearchStopReason.COVERAGE_SUFFICIENT
    assert len(state["research"]["search_queries"]) < state["research_control"].budget.max_search_queries
    assert [call["role"] for call in service.calls] == [LLMRole.COVERAGE_EVALUATOR, LLMRole.WRITER]
    assert "Minor long-term data limitation." in service.calls[1]["messages"][1].content
    assert service.calls[0]["metadata"].contains_private_data


@pytest.mark.parametrize("ids", [[], ["invented"], ["EV-1", "invented"], ["EV-other-objective"]])
def test_supported_assessment_requires_valid_references_for_its_own_objective(ids):
    state = initial_state()
    state["research"]["evidence"].append(state["research"]["evidence"][0].model_copy(update={
        "id": "EV-other-objective", "subquestion_id": "SQ-other",
    }))
    result = normalize(state, CoverageResult(status="sufficient", assessments=[assessment(ids=ids)]))

    # Readiness needs at least one valid reference; invalid ones only downgrade the assessment.
    expected = CoverageStatus.SUFFICIENT if "EV-1" in ids else CoverageStatus.INSUFFICIENT
    assert result.status == expected
    assert result.assessments[0].status == SubquestionCoverageStatus.PARTIAL
    assert set(result.assessments[0].evidence_ids).issubset({"EV-1"})
    assert not result.gaps  # Reference defects are not invented research questions.


@pytest.mark.parametrize("assessments", [[], [assessment(), assessment()]])
def test_missing_or_duplicate_assessments_cannot_confirm_readiness(assessments):
    result = normalize(initial_state(), CoverageResult(status="sufficient", assessments=assessments))
    assert result.status == CoverageStatus.INSUFFICIENT
    assert not result.gaps


def test_empty_evidence_cannot_be_normalized_into_readiness():
    state = initial_state()
    state["research"]["evidence"] = []
    result = normalize(state, CoverageResult(status="sufficient", assessments=[assessment()]))
    assert result.status == CoverageStatus.INSUFFICIENT
    assert result.assessments[0].status == SubquestionCoverageStatus.MISSING
    assert not result.assessments[0].evidence_ids


def test_no_evidence_skips_llm_and_retains_missing_objective_gaps():
    state = initial_state()
    state["research"]["evidence"] = []
    service = FakeLLM()
    result = asyncio.run(research_coverage_evaluator_node(state, llm_service=service))

    assert not service.calls
    assert result["research"]["iteration_count"] == 1
    assert result["research"]["coverage"].status == CoverageStatus.INSUFFICIENT
    assert result["research"]["coverage"].gaps[0].subquestion_id == "SQ-1"


def test_unsupported_overall_judgment_does_not_manufacture_a_gap_to_spend_more_budget():
    state = initial_state()
    result = normalize(state, CoverageResult(status="insufficient", assessments=[assessment()]))
    # Without a blocking gap the model's "insufficient" label does not trigger more research.
    assert result.status == CoverageStatus.SUFFICIENT
    assert result.assessments[0].status == SubquestionCoverageStatus.SUPPORTED
    assert not result.gaps
    state["research"]["coverage"] = result
    state["research"]["iteration_count"] = 1
    assert route(state) == "write_report"
    assert state["research"]["stop_reason"] == ResearchStopReason.COVERAGE_SUFFICIENT


def test_invalid_or_duplicate_gaps_are_filtered_without_forcing_partial_status():
    state = initial_state()
    result = normalize(state, CoverageResult(
        status="sufficient", assessments=[assessment()], gaps=[
            CoverageGap(subquestion_id="SQ-1", description=" Minor omission. "),
            CoverageGap(subquestion_id="SQ-1", description="minor omission."),
            CoverageGap(subquestion_id="unknown", description="Invented objective."),
            CoverageGap(subquestion_id="SQ-1", description=" "),
        ],
    ))
    assert result.status == CoverageStatus.SUFFICIENT
    assert [gap.description for gap in result.gaps] == ["Minor omission."]
    assert result.assessments[0].status == SubquestionCoverageStatus.SUPPORTED


def test_failed_reference_validation_logs_rejected_readiness(caplog):
    service = FakeLLM(CoverageResult(status="sufficient", assessments=[assessment(ids=["invented"])]))
    result = asyncio.run(research_coverage_evaluator_node(initial_state(), llm_service=service))
    assert result["research"]["coverage"].status == CoverageStatus.INSUFFICIENT
    record = next(item for item in caplog.records if getattr(item, "event", "") == "coverage_status_overridden")
    assert record.job_id == "job-1"
    assert record.iteration == 1


def test_provider_failure_is_terminal_with_iteration_and_log_context(caplog):
    service = FakeLLM(LLMServiceError(role=LLMRole.COVERAGE_EVALUATOR, message="Provider unavailable"))
    result = asyncio.run(research_coverage_evaluator_node(initial_state(), llm_service=service))
    assert result["research"]["iteration_count"] == 1
    assert result["errors"][0].component == "research_coverage_evaluator"
    assert not result["errors"][0].recoverable
    record = next(item for item in caplog.records if getattr(item, "event", "") == "coverage_evaluation_failed")
    assert record.job_id == "job-1"


def test_empty_plan_is_a_terminal_error_without_an_llm_call():
    state = initial_state()
    state["research"]["subquestions"] = []
    service = FakeLLM()
    result = asyncio.run(research_coverage_evaluator_node(state, llm_service=service))
    assert not service.calls
    assert not result["errors"][0].recoverable


def test_grouping_deduplicates_known_ids_and_ignores_orphan_evidence():
    state = initial_state()
    evidence = state["research"]["evidence"][0]
    result = _group_evidence_by_subquestion(
        subquestions=state["research"]["subquestions"],
        evidence=[evidence, evidence, evidence.model_copy(update={"id": "orphan", "subquestion_id": "unknown"})],
    )
    assert result == {"SQ-1": [evidence]}
