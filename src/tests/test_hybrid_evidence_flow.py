"""Focused offline tests for document and hybrid evidence flow."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace

from src.agents.research.research_evidence_extractor_node import (
    research_evidence_extractor_node,
)
from src.agents.research.research_retrieval_node import research_retrieval_node
from src.agents.writer.writer_citation_checker_node import _build_report_sources
from src.graph.graph_state import create_initial_workflow_state
from src.models import (
    EvidenceCandidate,
    Evidence,
    EvidenceExtractionOutput,
    ResearchSource,
    ResearchSourceMode,
    ReportFinding,
    RetrievalChunk,
    RetrievalResult,
    RetrievedChunk,
    SearchQuery,
    SearchResult,
    Source,
    SourceType,
    SubQuestion,
)
from src.services.research_job_resources import ResearchJobResources


def test_evidence_output_accepts_provider_items_wrapper():
    output = EvidenceExtractionOutput.model_validate({
        "evidence": {
            "items": [{
                "chunk_id": "chunk-1",
                "content": "Supported fact.",
                "relevance_score": 0.9,
            }],
        },
    })

    assert [item.chunk_id for item in output.evidence] == ["chunk-1"]


class FakeRetrievalSession:
    """Return deterministic web or document chunks."""

    def __init__(self, source_type: SourceType) -> None:
        self.source_type = source_type
        self.documents = []
        self.requests = []
        self.clear_calls = 0

    def build(self, documents) -> None:
        self.documents = list(documents)

    def retrieve(self, request) -> RetrievalResult:
        self.requests.append(request)
        if self.source_type == SourceType.WEB:
            document = self.documents[0]
            chunk = RetrievalChunk(
                chunk_id=f"{document.document_id}:0",
                document_id=document.document_id,
                content="Current web evidence",
                source_type="web",
                chunk_index=0,
                token_count=4,
                title=document.title,
                metadata=document.metadata,
            )
        else:
            chunk = RetrievalChunk(
                chunk_id="doc-1:block:0:0",
                document_id="doc-1:block:0",
                content="Private document evidence",
                source_type="document",
                chunk_index=0,
                token_count=4,
                title="report.pdf",
                section="Overview",
                metadata={
                    "owner_id": "owner-1",
                    "private_document_id": "doc-1",
                    "filename": "report.pdf",
                    "page_start": 2,
                    "page_end": 2,
                    "char_start": 20,
                    "char_end": 45,
                },
            )

        retrieved = RetrievedChunk(chunk=chunk, rank=1)
        return RetrievalResult(
            subquestion_id=request.subquestion_id,
            retrieval_query=request.subquestion,
            chunks=[retrieved],
            context=chunk.content,
            estimated_tokens=4,
        )

    def clear(self) -> None:
        self.clear_calls += 1


class FakeSessionFactory:
    def __init__(self) -> None:
        self.sessions = []

    def create_session(self) -> FakeRetrievalSession:
        session = FakeRetrievalSession(SourceType.WEB)
        self.sessions.append(session)
        return session


class FakeEvidenceLLM:
    def __init__(self, outputs) -> None:
        self.outputs = list(outputs)

    async def invoke_structured(self, **kwargs):
        return self.outputs.pop(0)


def create_hybrid_state():
    state = create_initial_workflow_state(
        job_id="job-1",
        research_session_id="session-1",
        question="Compare the report with current information",
        allowed_sources=[ResearchSource.WEB, ResearchSource.DOCUMENTS],
        source_mode=ResearchSourceMode.HYBRID,
        owner_id="owner-1",
        selected_document_ids=["doc-1"],
    )
    state["research"]["subquestions"] = [
        SubQuestion(id="sq-1", question="What changed?", objective="Compare")
    ]
    state["research"]["search_queries"] = [
        SearchQuery(
            id="query-1",
            subquestion_id="sq-1",
            query="current changes",
            purpose="Find changes",
            iteration=1,
        )
    ]
    state["research"]["search_results"] = [
        SearchResult(
            id="result-1",
            query_id="query-1",
            title="Web report",
            url="https://example.com/report",
        )
    ]
    state["research"]["sources"] = [
        Source(
            id="web-1",
            source_type=SourceType.WEB,
            search_result_ids=["result-1"],
            title="Web report",
            uri="https://example.com/report",
            content="Complete extracted webpage",
            retrieved_at=datetime.now(UTC),
        )
    ]
    return state


def test_hybrid_retrieval_keeps_results_separate_and_preserves_sources():
    async def run() -> None:
        state = create_hybrid_state()
        factory = FakeSessionFactory()
        document_session = FakeRetrievalSession(SourceType.DOCUMENT)
        resources = ResearchJobResources(document_session)

        result = await research_retrieval_node(
            state,
            {"configurable": {"job_resources": resources}},
            retrieval_session_factory=factory,
        )
        research = result["research"]

        assert len(research["web_retrieval_results"]) == 1
        assert len(research["document_retrieval_results"]) == 1
        assert factory.sessions[0].clear_calls == 1
        assert document_session.clear_calls == 0

        document_source = next(
            source for source in research["sources"]
            if source.source_type == SourceType.DOCUMENT
        )
        assert document_source.id == "doc-1"
        assert document_source.title == "report.pdf"

    asyncio.run(run())


def test_evidence_boundary_merges_web_and_document_provenance():
    async def run() -> None:
        state = create_hybrid_state()
        document_session = FakeRetrievalSession(SourceType.DOCUMENT)
        retrieval = await research_retrieval_node(
            state,
            {
                "configurable": {
                    "job_resources": ResearchJobResources(document_session)
                }
            },
            retrieval_session_factory=FakeSessionFactory(),
        )
        state["research"] = retrieval["research"]

        llm = FakeEvidenceLLM([
            EvidenceExtractionOutput(evidence=[
                EvidenceCandidate(
                    chunk_id="web-1:0",
                    content="Web finding",
                    relevance_score=0.9,
                )
            ]),
            EvidenceExtractionOutput(evidence=[
                EvidenceCandidate(
                    chunk_id="doc-1:block:0:0",
                    content="Document finding",
                    relevance_score=0.95,
                )
            ]),
        ])
        result = await research_evidence_extractor_node(
            state,
            llm_service=llm,
            settings=SimpleNamespace(
                research_limits=SimpleNamespace(
                    max_evidence_items=10,
                    min_evidence_relevance=0.5,
                )
            ),
        )
        evidence = result["research"]["evidence"]

        assert [item.source_type for item in evidence] == [
            SourceType.WEB,
            SourceType.DOCUMENT,
        ]
        assert evidence[1].source_id == "doc-1"
        assert evidence[1].chunk_id == "doc-1:block:0:0"
        assert evidence[1].provenance["filename"] == "report.pdf"
        assert evidence[1].provenance["page_start"] == 2
        assert {item.source_id for item in evidence}.issubset(
            {source.id for source in state["research"]["sources"]}
        )

    asyncio.run(run())


def test_document_session_is_reused_across_coverage_iterations():
    async def run() -> None:
        state = create_hybrid_state()
        state["allowed_sources"] = [ResearchSource.DOCUMENTS]
        document_session = FakeRetrievalSession(SourceType.DOCUMENT)
        resources = ResearchJobResources(document_session)
        factory = FakeSessionFactory()

        await research_retrieval_node(
            state,
            {"configurable": {"job_resources": resources}},
            retrieval_session_factory=factory,
        )
        state["research"]["iteration_count"] = 1
        state["research"]["search_queries"].append(
            SearchQuery(
                id="query-2",
                subquestion_id="sq-1",
                query="remaining gap",
                purpose="Close gap",
                iteration=2,
            )
        )
        await research_retrieval_node(
            state,
            {"configurable": {"job_resources": resources}},
            retrieval_session_factory=factory,
        )

        assert len(document_session.requests) == 2
        assert document_session.clear_calls == 0
        assert factory.sessions == []

    asyncio.run(run())


def test_hybrid_evidence_deduplicates_matching_claims_across_sources():
    async def run() -> None:
        state = create_hybrid_state()
        retrieval = await research_retrieval_node(
            state,
            {
                "configurable": {
                    "job_resources": ResearchJobResources(
                        FakeRetrievalSession(SourceType.DOCUMENT)
                    )
                }
            },
            retrieval_session_factory=FakeSessionFactory(),
        )
        state["research"] = retrieval["research"]
        shared_content = "The same supported claim"
        llm = FakeEvidenceLLM([
            EvidenceExtractionOutput(evidence=[EvidenceCandidate(
                chunk_id="web-1:0",
                content=shared_content,
                relevance_score=0.9,
            )]),
            EvidenceExtractionOutput(evidence=[EvidenceCandidate(
                chunk_id="doc-1:block:0:0",
                content=shared_content,
                relevance_score=0.9,
            )]),
        ])

        result = await research_evidence_extractor_node(
            state,
            llm_service=llm,
            settings=SimpleNamespace(
                research_limits=SimpleNamespace(
                    max_evidence_items=10,
                    min_evidence_relevance=0.5,
                )
            ),
        )

        assert len(result["research"]["evidence"]) == 1

    asyncio.run(run())


def test_document_report_source_exposes_safe_locations():
    source = Source(
        id="doc-1",
        source_type=SourceType.DOCUMENT,
        title="report.pdf",
        uri="document://doc-1",
        content="Retrieved passage",
        retrieved_at=datetime.now(UTC),
    )
    evidence = Evidence(
        id="evidence-1",
        subquestion_id="sq-1",
        source_id="doc-1",
        chunk_id="doc-1:block:0:0",
        source_type=SourceType.DOCUMENT,
        provenance={
            "owner_id": "owner-1",
            "filename": "report.pdf",
            "page_start": 2,
            "section": "Overview",
        },
        content="Supported claim",
        relevance_score=0.9,
    )

    report_sources = _build_report_sources(
        findings=[ReportFinding(
            title="Finding",
            content="Supported claim",
            evidence_ids=["evidence-1"],
        )],
        evidence_by_id={evidence.id: evidence},
        sources_by_id={source.id: source},
    )

    assert report_sources[0].locations == [{
        "filename": "report.pdf",
        "page_start": 2,
        "section": "Overview",
        "chunk_id": "doc-1:block:0:0",
    }]
    assert "owner_id" not in report_sources[0].locations[0]
