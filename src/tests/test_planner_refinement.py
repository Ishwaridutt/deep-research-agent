"""Offline regression tests for planner-owned research refinement."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from src.agents.research.research_planner_node import _build_subquestions, _refinement_context, research_planner_node
from src.agents.research.research_query_generator_node import research_query_generator_node
from src.agents.research.research_retrieval_node import _retrieval_requests
from src.graph import graph_builder
from src.graph.graph_routing import route_after_planning
from src.graph.graph_state import create_initial_workflow_state
from src.llm import LLMRole, LLMServiceError
from src.models import (
    CoverageGap, CoverageResult, CoverageStatus, Evidence, EvidenceCandidate,
    EvidenceExtractionOutput, FollowupPlannerOutput, GeneratedSearchQuery,
    PlannedFollowupSubQuestion, PlannedSubQuestion, PlannerOutput,
    QueryGeneratorOutput, ResearchSource, ResearchStopReason,
    RetrievalChunk, RetrievalRequest, RetrievalResult, RetrievedChunk,
    SearchQuery, SourceType, SubQuestion, SubquestionSearchQueries,
)
from src.models.coverage_model import EvidenceStrength, SubquestionCoverage, SubquestionCoverageStatus
from src.retrieval.context.retrieval_context_builder import RetrievalContextBuilder
from src.retrieval.retrieval_session import RetrievalSession


def settings(**overrides):
    limits = dict(
        max_subquestions=5, max_queries_per_subquestion=5,
        max_total_queries_per_job=30,
        max_search_query_chars=500, max_iterations=3,
        max_evidence_items=100, min_evidence_relevance=0.6,
    )
    return SimpleNamespace(
        research_limits=SimpleNamespace(**(limits | overrides)),
        search=SimpleNamespace(extraction_max_concurrency=5),
    )


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
        output = next(self.outputs)
        # Callables build outputs from the prompt when subquestion IDs are generated at runtime.
        return output(**kwargs) if callable(output) else output


def query_output(groups):
    """Build one batched query generator output from {subquestion_id: [query, ...]}."""
    return QueryGeneratorOutput(subquestions=[
        SubquestionSearchQueries(subquestion_id=subquestion_id, queries=[
            GeneratedSearchQuery(query=query, purpose="Measure") for query in queries
        ])
        for subquestion_id, queries in groups.items()
    ])


def prompt_subquestion_ids(messages):
    """Read the subquestion IDs supplied in a query generator prompt."""
    tasks = json.loads(messages[1].content.split("<subquestions>\n")[1].split("\n</subquestions>")[0])
    return [task["subquestion_id"] for task in tasks]


def test_refinement_context_keeps_task_history_and_gaps_without_evidence():
    state = followup_state()
    previous = SubQuestion(
        id="previous-followup", parent_subquestion_id="root", iteration=2,
        question="What measurements are reported?", objective="Find measured outcomes.",
    )
    state["research"]["planning_history"].append(previous)
    state["research"]["search_queries"].append(SearchQuery(
        id="followup-query", subquestion_id=previous.id, iteration=2,
        query="measured outcomes trial", purpose="Retrieve missing results.",
    ))
    low_priority = CoverageGap(subquestion_id="root", description="Missing comparison.", priority=2)
    state["research_control"].pending_gaps.insert(0, low_priority)
    state["research"]["coverage"] = CoverageResult(
        status=CoverageStatus.INSUFFICIENT, gaps=state["research_control"].pending_gaps,
    )
    state["research"]["evidence"] = [Evidence(
        id="evidence-1", subquestion_id="root", source_id="source-1", chunk_id="chunk-1",
        source_type=SourceType.WEB, content="Evidence body stays outside planner context.", relevance_score=0.9,
    )]

    context = _refinement_context(state, available_query_slots=3, max_queries_per_subquestion=2)

    assert set(context) == {
        "original_subquestions", "previous_tasks", "unresolved_gaps",
        "available_query_slots", "max_queries_per_subquestion",
    }
    assert context["original_subquestions"][0]["objective"] == "Assess the outcome."
    assert [item["queries"] for item in context["previous_tasks"]] == [
        ["original search"], ["measured outcomes trial"],
    ]
    assert context["previous_tasks"][1]["parent_subquestion_id"] == "root"
    assert [gap["priority"] for gap in context["unresolved_gaps"]] == [5, 2]
    assert context["available_query_slots"] == 3
    assert context["max_queries_per_subquestion"] == 2
    assert "Evidence body" not in json.dumps(context)
    assert len(state["research"]["evidence"]) == 1
    assert state["research"]["coverage"] is not None


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


def test_query_allocation_spreads_budget_and_rejects_old_queries():
    state = followup_state()
    state["research"]["active_subquestions"] = [
        SubQuestion(id=f"followup-{i}", parent_subquestion_id="root", iteration=2,
                    question=f"Find missing facet {i}?", objective="Collect measurements.")
        for i in range(2)
    ]
    llm = FakeLLM([query_output({
        "followup-0": ["first facet"],
        "followup-1": [" ORIGINAL SEARCH ", "first facet", *[f"second facet {i}" for i in range(6)]],
    })])
    update = asyncio.run(research_query_generator_node(
        state, llm_service=llm, settings=settings(max_total_queries_per_job=6),
    ))
    new_queries = update["research"]["search_queries"][1:]
    # One batched call; five remaining slots spread as at most two per task.
    assert len(llm.calls) == 1
    assert "no more than 2" in llm.calls[0]["messages"][0].content
    assert "original search" in llm.calls[0]["messages"][1].content
    assert [query.query for query in new_queries] == ["first facet", "second facet 0", "second facet 1"]
    assert all(query.iteration == 2 and query.providers == ["web"] for query in new_queries)


def test_query_allocation_reserves_one_for_each_selected_task():
    state = followup_state()
    state["research"]["active_subquestions"] = [
        SubQuestion(id=f"task-{i}", parent_subquestion_id="root", question=f"Facet {i}?", objective="Measure.")
        for i in range(5)
    ]
    llm = FakeLLM([query_output({f"task-{i}": [f"query {i}", f"extra {i}"] for i in range(5)})])
    update = asyncio.run(research_query_generator_node(
        state, llm_service=llm, settings=settings(max_total_queries_per_job=6),
    ))
    assert len(update["research"]["search_queries"]) == 6
    assert len(llm.calls) == 1
    assert "no more than 1" in llm.calls[0]["messages"][0].content


def test_query_counts_follow_planned_tasks_across_rounds():
    def batched(prefix):
        def build(*, messages, **kwargs):
            return query_output({
                subquestion_id: [f"{prefix} aspect {i} evidence {j}" for j in range(2)]
                for i, subquestion_id in enumerate(prompt_subquestion_ids(messages))
            })
        return build

    async def run():
        state = initial_state()
        limits = settings(max_queries_per_subquestion=2)
        initial_llm = FakeLLM([
            PlannerOutput(subquestions=[PlannedSubQuestion(
                question=f"What explains aspect {i}?", objective=f"Research aspect {i}.",
            ) for i in range(5)]),
            batched("initial"),
        ])
        state.update(await research_planner_node(state, llm_service=initial_llm, settings=limits))
        state.update(await research_query_generator_node(state, llm_service=initial_llm, settings=limits))
        assert len(state["research"]["subquestions"]) == 5
        assert len(state["research"]["search_queries"]) == 10

        roots = state["research"]["subquestions"]
        state["research"]["iteration_count"] = 1
        state["research_control"].pending_gaps = [CoverageGap(
            subquestion_id=root.id, description=f"Missing outcome {j} for {root.question}", blocking=True,
        ) for root in roots[:3] for j in range(2)]
        followup_llm = FakeLLM([
            FollowupPlannerOutput(subquestions=[PlannedFollowupSubQuestion(
                parent_subquestion_id=root.id, question=f"Which outcomes are missing for aspect {i}?",
                objective="Gather the two related missing outcomes.",
            ) for i, root in enumerate(roots[:3])]),
            batched("followup"),
        ])
        state.update(await research_planner_node(state, llm_service=followup_llm, settings=limits))
        state.update(await research_query_generator_node(state, llm_service=followup_llm, settings=limits))

        assert len(state["research"]["active_subquestions"]) == 3
        assert len(state["research"]["subquestions"]) == 5
        queries = state["research"]["search_queries"]
        assert len([query for query in queries if query.iteration == 2]) == 6
        assert len(queries) == 16
        prompt = followup_llm.calls[0]["messages"][1].content
        context = json.loads(prompt.split("<refinement_context>\n")[1].split("\n</refinement_context>")[0])
        assert context["available_query_slots"] == 20
        assert len(context["unresolved_gaps"]) == 6

    asyncio.run(run())


def test_per_subquestion_query_limit_still_applies_without_round_cap():
    state = followup_state()
    llm = FakeLLM([query_output({"root": [f"new evidence {i}" for i in range(6)]})])
    result = asyncio.run(research_query_generator_node(
        state, llm_service=llm, settings=settings(max_queries_per_subquestion=2),
    ))
    assert len(result["research"]["search_queries"]) == 3  # One previous query plus two new queries.


def test_long_queries_are_shortened_and_native_providers_keep_web():
    state = followup_state()
    long_query = " ".join(f"word{i}" for i in range(20))
    llm = FakeLLM([QueryGeneratorOutput(subquestions=[SubquestionSearchQueries(subquestion_id="root", queries=[
        GeneratedSearchQuery(query=long_query, purpose="Find papers", providers=["arxiv"]),
    ])])])
    result = asyncio.run(research_query_generator_node(
        state, llm_service=llm, settings=settings(max_search_query_words=12),
    ))
    query = result["research"]["search_queries"][-1]
    assert query.query == " ".join(f"word{i}" for i in range(12))
    assert query.providers == ["web", "arxiv"]


def test_no_unique_queries_goes_to_writer_without_an_empty_search_round():
    state = followup_state()
    llm = FakeLLM([query_output({"root": ["ORIGINAL SEARCH"]})])
    update = asyncio.run(research_query_generator_node(state, llm_service=llm, settings=settings()))
    state.update(update)
    assert route_after_planning(state) == "write_report"
    assert len(state["research"]["search_queries"]) == 1


def test_query_budget_exhaustion_skips_llm_and_records_budget_stop():
    for executed_queries, job_limit in [(30, 30), (0, 1)]:
        state = followup_state()
        state["research_control"].usage.queries_executed = executed_queries
        state["research_control"].budget.max_search_queries = job_limit
        llm = FakeLLM([])
        result = asyncio.run(research_query_generator_node(state, llm_service=llm, settings=settings()))
        assert llm.calls == []
        assert result["research"]["stop_reason"] == ResearchStopReason.QUERY_BUDGET_EXHAUSTED
        assert result["research_control"].last_decision.budget_exhausted
        assert result["errors"] == []


def test_query_generation_skips_expired_jobs():
    state = followup_state()
    state["research_control"].usage.started_at = datetime.now(UTC) - timedelta(hours=1)
    llm = FakeLLM([])
    result = asyncio.run(research_query_generator_node(state, llm_service=llm, settings=settings()))
    assert llm.calls == []
    assert result["research"]["stop_reason"] == ResearchStopReason.TIME_BUDGET_EXHAUSTED
    assert result["research"]["search_queries"] == state["research"]["search_queries"]


def test_query_timeout_stops_the_round_without_new_queries():
    state = followup_state()
    state["allowed_sources"].append(ResearchSource.DOCUMENTS)
    state["research"]["active_subquestions"] = [SubQuestion(
        id=f"task-{i}", question=f"Question {i}?", objective="Find evidence.",
        parent_subquestion_id="root",
    ) for i in range(3)]

    class TimeoutLLM:
        def __init__(self):
            self.calls = []

        async def invoke_structured(self, **kwargs):
            self.calls.append(kwargs)
            raise TimeoutError

    llm = TimeoutLLM()
    result = asyncio.run(research_query_generator_node(state, llm_service=llm, settings=settings()))
    assert len(llm.calls) == 1
    assert llm.calls[0]["metadata"].contains_private_data
    assert result["research"]["stop_reason"] == ResearchStopReason.TIME_BUDGET_EXHAUSTED
    assert [query.query for query in result["research"]["search_queries"]] == ["original search"]
    assert result["research_control"].last_decision.action == "write_report"


def test_failed_generation_falls_back_to_subquestion_text():
    state = followup_state()
    state["research"]["active_subquestions"] = [SubQuestion(
        id=f"task-{i}", question=f"Question {i}?", objective="Find evidence.",
    ) for i in range(2)]

    class FailingLLM:
        async def invoke_structured(self, **kwargs):
            raise LLMServiceError(role=LLMRole.QUERY_GENERATOR, message="Provider unavailable.")

    result = asyncio.run(research_query_generator_node(state, llm_service=FailingLLM(), settings=settings()))
    assert [query.query for query in result["research"]["search_queries"][1:]] == ["Question 0?", "Question 1?"]
    assert result["research"]["stop_reason"] is None
    assert len(result["errors"]) == 1 and result["errors"][0].recoverable


def test_invalid_queries_are_rejected_and_skipped_tasks_get_a_fallback():
    state = followup_state()
    state["research"]["active_subquestions"] = [SubQuestion(
        id=f"task-{i}", question=f"Question {i}?", objective="Find evidence.",
    ) for i in range(2)]
    llm = FakeLLM([QueryGeneratorOutput(subquestions=[
        SubquestionSearchQueries(subquestion_id="task-1", queries=[
            GeneratedSearchQuery(query=query, purpose=purpose) for query, purpose in [
                (" ", "Find evidence"), ("blank purpose", " "), ("x" * 50, "Find evidence"),
                (" ORIGINAL   SEARCH ", "Already attempted"), ("valid search", "Find evidence"),
            ]
        ]),
        SubquestionSearchQueries(subquestion_id="unknown", queries=[
            GeneratedSearchQuery(query="invented task", purpose="Find evidence"),
        ]),
    ])])
    result = asyncio.run(research_query_generator_node(
        state, llm_service=llm, settings=settings(max_search_query_chars=20),
    ))
    queries = result["research"]["search_queries"][1:]
    assert [(query.subquestion_id, query.query) for query in queries] == [
        ("task-0", "Question 0?"), ("task-1", "valid search"),
    ]
    assert result["research"]["stop_reason"] is None
    assert result["errors"] == []


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
            return {"Compare both facets.": [], "intent A": [common, first], "intent B": [second, common]}[query]

    retriever = Retriever()
    session = RetrievalSession(
        chunker=None, hybrid_retriever=retriever,
        context_builder=RetrievalContextBuilder(len, max_context_tokens=2000),
    )
    result = session.retrieve(RetrievalRequest(
        subquestion_id="followup", research_question="Explain the system.",
        subquestion="Compare both facets.", search_queries=["intent A", "intent B", "intent A"],
    ))
    # The subquestion always leads as its own retrieval intent.
    assert retriever.calls == ["Compare both facets.", "intent A", "intent B"]
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
                return query_output({
                    subquestion_id: [f"measured outcome {iteration}"]
                    for subquestion_id in prompt_subquestion_ids(messages)
                })
            if role == LLMRole.EVIDENCE_EXTRACTOR:
                return EvidenceExtractionOutput(evidence=[EvidenceCandidate(
                    chunk_id=f"chunk-{iteration}", relevance_score=0.9,
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
                    blocking=True,
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
    assert all("coverage" not in context and "accepted_findings" not in context for context in planner_inputs)
    assert planner_inputs[1]["unresolved_gaps"][0]["subquestion_id"] == root_id
    assert len(final["research_control"].history) == 3


def test_query_helpers_clean_select_and_fall_back_without_the_node():
    from src.agents.research.research_query_generator_node import _clean_query, _select_queries

    assert _clean_query("  ", max_words=12, max_chars=500) is None
    assert _clean_query("x" * 30, max_words=12, max_chars=20) is None
    assert _clean_query("one two three four", max_words=2, max_chars=500) == "one two"

    tasks = [SubQuestion(id=f"t{i}", question=f"Question {i}?", objective="Find.") for i in range(3)]
    generated_queries = QueryGeneratorOutput(subquestions=[
        SubquestionSearchQueries(subquestion_id="t0", queries=[
            GeneratedSearchQuery(query="first", purpose="Find"), GeneratedSearchQuery(query="second", purpose="Find"),
        ]),
    ])
    previous = [SearchQuery(id="old", subquestion_id="t1", query="Question 1?", purpose="Old", iteration=1)]
    queries, fallback_ids = _select_queries(
        subquestions=tasks, generated_queries=generated_queries, previous_queries=previous, budget=3,
        per_subquestion=2, max_words=12, max_chars=500, iteration=2,
    )
    # t0 uses the model's queries; t1's fallback duplicates an old search; t2 falls back to its text.
    assert [(query.subquestion_id, query.query) for query in queries] == [
        ("t0", "first"), ("t0", "second"), ("t2", "Question 2?"),
    ]
    assert fallback_ids == ["t2"]
    assert all(query.iteration == 2 and query.providers == ["web"] for query in queries)
