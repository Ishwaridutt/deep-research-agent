"""Offline regression tests for planner-owned research refinement."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from src.agents.research.research_planner_node import _build_subquestions, research_planner_node
from src.agents.research.research_query_generator_node import research_query_generator_node
from src.agents.research.research_retrieval_node import _retrieval_requests
from src.graph import graph_builder
from src.graph.graph_routing import route_after_planning
from src.graph.graph_state import create_initial_workflow_state
from src.llm import LLMRole
from src.models import (
    CoverageGap, CoverageResult, CoverageStatus, EvidenceCandidate,
    EvidenceExtractionOutput, FollowupPlannerOutput, GeneratedSearchQuery,
    PlannedFollowupSubQuestion, PlannedSubQuestion, PlannerOutput,
    QueryGeneratorOutput, ResearchSource, ResearchStopReason,
    RetrievalChunk, RetrievalRequest, RetrievalResult, RetrievedChunk,
    SearchQuery, SubQuestion,
)
from src.models.coverage_model import EvidenceStrength, SubquestionCoverage, SubquestionCoverageStatus
from src.retrieval.context.retrieval_context_builder import RetrievalContextBuilder
from src.retrieval.retrieval_session import RetrievalSession


def settings(**overrides):
    limits = dict(
        max_subquestions=5, max_queries_per_subquestion=5,
        max_followup_queries_per_round=5, max_total_queries_per_job=30,
        max_search_query_chars=500, max_iterations=3,
        max_evidence_items=100, min_evidence_relevance=0.6,
    )
    return SimpleNamespace(research_limits=SimpleNamespace(**(limits | overrides)))


def initial_state():
    return create_initial_workflow_state(
        job_id="job-1", research_session_id="session-1",
        question="Explain how the system works.", allowed_sources=[ResearchSource.WEB],
    )


def followup_state():
    state = initial_state()
    root = SubQuestion(id="root", question="What affects the outcome?", objective="Assess the outcome.")
    state["research"].update(subquestions=[root], planning_history=[root], iteration_count=1)
    state["research_control"].pending_gaps = [
        CoverageGap(subquestion_id="root", description="Missing measurements.", priority=5),
    ]
    state["research"]["search_queries"] = [SearchQuery(
        id="old-query", subquestion_id="root", query="original search",
        purpose="Initial evidence", iteration=1,
    )]
    return state


class FakeLLM:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.calls = []

    async def invoke_structured(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.outputs)


def test_followup_validation_rejects_unknown_parents_and_repeated_tasks():
    previous = [SubQuestion(id="root", question="Existing task?", objective="Collect evidence.")]
    output = FollowupPlannerOutput(subquestions=[
        PlannedFollowupSubQuestion(parent_subquestion_id=parent, question=question, objective="Find missing data.")
        for parent, question in [
            ("unknown", "Unrelated task?"), ("root", "  EXISTING task? "),
            ("root", "What is the measured effect?"), ("root", "Another question?"),
        ]
    ])
    accepted = _build_subquestions(
        planner_output=output, max_subquestions=1, previous_subquestions=previous,
        allowed_parent_ids={"root"}, iteration=2,
    )
    assert len(accepted) == 1
    assert accepted[0].question == "What is the measured effect?"
    assert accepted[0].parent_subquestion_id == "root"
    assert accepted[0].iteration == 2


def test_empty_followup_goes_to_writer_without_replacing_original_objectives():
    state = followup_state()
    llm = FakeLLM([FollowupPlannerOutput(subquestions=[])])
    update = asyncio.run(research_planner_node(state, llm_service=llm, settings=settings()))
    state.update(update)
    assert state["research"]["subquestions"][0].id == "root"
    assert state["research"]["active_subquestions"] == []
    assert state["research"]["iteration_count"] == 1
    assert state["research"]["stop_reason"] == ResearchStopReason.NO_ACTIONABLE_GAPS
    assert route_after_planning(state) == "write_report"
    assert state["research_control"].last_decision.action == "write_report"
    prompt = llm.calls[0]["messages"][1].content
    assert "original search" in prompt and "Missing measurements" in prompt
    assert llm.calls[0]["output_schema"] is FollowupPlannerOutput


def test_budget_exhaustion_skips_refinement_llm():
    state = followup_state()
    llm = FakeLLM([])
    update = asyncio.run(research_planner_node(
        state, llm_service=llm, settings=settings(max_total_queries_per_job=1),
    ))
    assert update["research"]["stop_reason"] == ResearchStopReason.QUERY_BUDGET_EXHAUSTED
    assert llm.calls == []


def test_expired_research_budget_skips_planner_llm():
    state = followup_state()
    state["research_control"].usage.started_at = datetime.now(UTC) - timedelta(hours=1)
    llm = FakeLLM([])
    update = asyncio.run(research_planner_node(state, llm_service=llm, settings=settings()))
    assert update["research"]["stop_reason"] == ResearchStopReason.TIME_BUDGET_EXHAUSTED
    assert llm.calls == []


def test_retrieval_uses_active_followup_instead_of_original_objective():
    state = followup_state()
    followup = SubQuestion(
        id="followup", parent_subquestion_id="root", iteration=2,
        question="Which measured outcome is missing?", objective="Find measurements.",
    )
    state["research"]["active_subquestions"] = [followup]
    query = SearchQuery(
        id="new-query", subquestion_id="followup", iteration=2,
        query="targeted measurements", purpose="Resolve missing outcome.",
    )
    requests = _retrieval_requests(state, [query])
    assert len(requests) == 1
    assert requests[0].subquestion_id == "followup"
    assert requests[0].subquestion == followup.question
    assert requests[0].search_queries == [query.query]


def test_query_allocation_reuses_unused_slots_and_rejects_old_queries():
    state = followup_state()
    state["research"]["active_subquestions"] = [
        SubQuestion(id=f"followup-{i}", parent_subquestion_id="root", iteration=2,
                    question=f"Find missing facet {i}?", objective="Collect measurements.")
        for i in range(2)
    ]
    llm = FakeLLM([
        QueryGeneratorOutput(queries=[GeneratedSearchQuery(query="first facet", purpose="Measure")]),
        QueryGeneratorOutput(queries=[
            GeneratedSearchQuery(query=query, purpose="Measure")
            for query in [" ORIGINAL SEARCH ", "first facet", *[f"second facet {i}" for i in range(6)]]
        ]),
    ])
    update = asyncio.run(research_query_generator_node(state, llm_service=llm, settings=settings()))
    new_queries = update["research"]["search_queries"][1:]
    assert len(new_queries) == 5
    assert [query.subquestion_id for query in new_queries].count("followup-0") == 1
    assert [query.subquestion_id for query in new_queries].count("followup-1") == 4
    assert "no more than 4 queries" in llm.calls[1]["messages"][0].content
    assert "original search" in llm.calls[1]["messages"][1].content
    assert all(query.iteration == 2 for query in new_queries)


def test_query_allocation_reserves_one_for_each_selected_task():
    state = followup_state()
    state["research"]["active_subquestions"] = [
        SubQuestion(id=f"task-{i}", parent_subquestion_id="root", question=f"Facet {i}?", objective="Measure.")
        for i in range(5)
    ]
    llm = FakeLLM([
        QueryGeneratorOutput(queries=[GeneratedSearchQuery(query=f"query {i}", purpose="Measure")])
        for i in range(5)
    ])
    update = asyncio.run(research_query_generator_node(state, llm_service=llm, settings=settings()))
    assert len(update["research"]["search_queries"]) == 6
    assert all("no more than 1 queries" in call["messages"][0].content for call in llm.calls)


def test_no_unique_queries_goes_to_writer_without_an_empty_search_round():
    state = followup_state()
    llm = FakeLLM([QueryGeneratorOutput(queries=[
        GeneratedSearchQuery(query="ORIGINAL SEARCH", purpose="Already tried"),
    ])])
    update = asyncio.run(research_query_generator_node(state, llm_service=llm, settings=settings()))
    state.update(update)
    assert route_after_planning(state) == "write_report"
    assert len(state["research"]["search_queries"]) == 1


def test_multiquery_retrieval_searches_independently_and_merges_once():
    def chunk(identifier):
        return RetrievedChunk(RetrievalChunk(
            chunk_id=identifier, document_id=identifier, content=f"Evidence {identifier}",
            source_type="web", chunk_index=0, token_count=2,
        ), rank=1)

    common, first, second = chunk("common"), chunk("first"), chunk("second")

    class Retriever:
        def __init__(self):
            self.calls = []

        def search(self, query, top_k):
            self.calls.append(query)
            return {"intent A": [common, first], "intent B": [second, common]}[query]

    retriever = Retriever()
    session = RetrievalSession(
        chunker=None, hybrid_retriever=retriever,
        context_builder=RetrievalContextBuilder(len, max_context_tokens=2000),
    )
    result = session.retrieve(RetrievalRequest(
        subquestion_id="followup", research_question="Explain the system.",
        subquestion="Compare both facets.", search_queries=["intent A", "intent B", "intent A"],
    ))
    assert retriever.calls == ["intent A", "intent B"]
    assert [item.chunk.chunk_id for item in result.chunks] == ["common", "second", "first"]
    assert result.subquestion_id == "followup"
    assert result.estimated_tokens <= 2000


def test_real_graph_replans_twice_and_accumulates_evidence_under_original_objective(monkeypatch):
    coverage_inputs = []
    planner_inputs = []

    class ResearchLLM:
        async def invoke_structured(self, *, role, messages, metadata, **kwargs):
            iteration = metadata.iteration
            if role == LLMRole.PLANNER:
                if iteration == 1:
                    return PlannerOutput(subquestions=[PlannedSubQuestion(
                        question="What explains the outcome?", objective="Identify the causes.",
                    )])
                context = json.loads(messages[1].content.split("<refinement_context>\n")[1].split("\n</refinement_context>")[0])
                planner_inputs.append(context)
                return FollowupPlannerOutput(subquestions=[PlannedFollowupSubQuestion(
                    parent_subquestion_id=context["original_subquestions"][0]["id"],
                    question=f"What evidence explains missing outcome {iteration}?",
                    objective=f"Measure missing outcome {iteration}.",
                )])
            if role == LLMRole.QUERY_GENERATOR:
                return QueryGeneratorOutput(queries=[GeneratedSearchQuery(
                    query=f"measured outcome {iteration}", purpose="Find direct evidence.",
                )])
            if role == LLMRole.EVIDENCE_EXTRACTOR:
                return EvidenceExtractionOutput(evidence=[EvidenceCandidate(
                    chunk_id=f"chunk-{iteration}", content=f"Measured finding {iteration}.", relevance_score=0.9,
                )])
            assert role == LLMRole.COVERAGE_EVALUATOR
            context = json.loads(messages[1].content.split("<research_context>\n")[1].split("\n</research_context>")[0])
            objective = context["subquestions"][0]
            coverage_inputs.append(objective)
            supported = iteration == 3
            return CoverageResult(
                status=CoverageStatus.SUFFICIENT if supported else CoverageStatus.INSUFFICIENT,
                gaps=[] if supported else [CoverageGap(
                    subquestion_id=objective["subquestion_id"], description=f"Missing outcome {iteration + 1}.",
                )],
                assessments=[SubquestionCoverage(
                    subquestion_id=objective["subquestion_id"],
                    status=SubquestionCoverageStatus.SUPPORTED if supported else SubquestionCoverageStatus.PARTIAL,
                    evidence_strength=EvidenceStrength.MODERATE,
                    evidence_ids=[item["id"] for item in objective["evidence"]],
                    reason="Evidence addresses the objective." if supported else "Additional outcome remains missing.",
                )],
            )

    async def passthrough(state, **kwargs):
        return {}

    async def retrieve(state, **kwargs):
        iteration = state["research"]["iteration_count"] + 1
        active = state["research"]["active_subquestions"][0]
        chunk = RetrievalChunk(
            chunk_id=f"chunk-{iteration}", document_id=f"source-{iteration}",
            content=f"Measured finding {iteration}.", source_type="web", chunk_index=0, token_count=4,
        )
        return {"research": {**state["research"], "web_retrieval_results": [RetrievalResult(
            subquestion_id=active.id, retrieval_query=active.question,
            chunks=[RetrievedChunk(chunk, 1)], context=chunk.content, estimated_tokens=4,
        )]}}

    for name in [
        "input_guardrail_node", "research_source_selection_node", "research_web_search_node",
        "research_source_processor_node", "writer_prepare_evidence_node", "writer_synthesis_node",
        "writer_citation_checker_node", "finalize_report_node",
    ]:
        monkeypatch.setattr(graph_builder, name, passthrough)
    monkeypatch.setattr(graph_builder, "research_retrieval_node", retrieve)
    graph = graph_builder.build_research_graph(
        settings=settings(), llm_service=ResearchLLM(), search_provider=None,
        source_extractor=None, source_cleaner=None, retrieval_session_factory=None, document_service=None,
    )
    final = asyncio.run(graph.ainvoke(initial_state(), config={"recursion_limit": 60}))
    research = final["research"]
    root_id = research["subquestions"][0].id
    assert research["iteration_count"] == 3
    assert research["stop_reason"] == ResearchStopReason.COVERAGE_SUFFICIENT
    assert len(research["subquestions"]) == 1
    assert len(research["planning_history"]) == 3
    assert {item.subquestion_id for item in research["evidence"]} == {root_id}
    assert [len(item["evidence"]) for item in coverage_inputs] == [1, 2, 3]
    assert {item["subquestion_id"] for item in coverage_inputs} == {root_id}
    assert research["evidence"][2].provenance["research_subquestion_id"] == research["active_subquestions"][0].id
    assert len(planner_inputs[1]["previous_tasks"]) == 2
    assert planner_inputs[1]["previous_tasks"][1]["queries"] == ["measured outcome 2"]
    assert len(planner_inputs[1]["accepted_findings"][root_id]) == 2
    assert len(final["research_control"].history) == 3
