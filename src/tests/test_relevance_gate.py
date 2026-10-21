"""Offline tests for the pre-extraction relevance gate and parallel extraction."""

import asyncio

from src.agents.research.research_source_processor_node import research_source_processor_node
from src.graph.graph_state import create_initial_workflow_state
from src.models import ResearchSource, SearchQuery, SearchResult, SourceProvider, SubQuestion
from src.tools.cleaning import SourceCleaner
from src.tools.extraction import ExtractedSourceContent
from src.tools.search.relevance_gate import GateCandidate, SearchRelevanceGate, fair_order


class KeywordEmbeddings:
    """Fake normalized embeddings: one dimension per topic keyword."""

    TOPICS = ("muscle", "protein", "cosmology")

    def _vector(self, text):
        hits = [float(topic in text.casefold()) for topic in self.TOPICS]
        norm = sum(value * value for value in hits) ** 0.5 or 1.0
        return [value / norm for value in hits]

    def embed_query(self, text):
        return self._vector(text)

    def embed_documents(self, texts):
        return [self._vector(text) for text in texts]


def test_gate_scores_by_meaning_keeps_minimum_and_orders_fairly():
    gate = SearchRelevanceGate(KeywordEmbeddings(), threshold=0.7, min_keep=1)
    scores = gate.evaluate(
        subquestions={"sq-muscle": "How to build muscle?", "sq-protein": "How much protein?"},
        candidates=[
            GateCandidate("muscle-page", "Muscle growth guide", ("sq-muscle",)),
            GateCandidate("cosmology-paper", "Fat convex sets in cosmology", ("sq-muscle",)),
            GateCandidate("protein-page", "Protein and muscle intake", ("sq-protein",)),
            GateCandidate("carried", "Muscle training volume", ()),
        ],
    )
    by_key = {item.key: item for item in scores}
    assert by_key["muscle-page"].decision == "passed" and by_key["muscle-page"].score == 1.0
    assert by_key["cosmology-paper"].decision == "rejected" and by_key["cosmology-paper"].score == 0.0
    # 0.707 similarity passes the 0.7 gate.
    assert by_key["protein-page"].decision == "passed"
    # Carried candidates are matched against every active subquestion.
    assert by_key["carried"].subquestion_id == "sq-muscle"
    ordered = [item.key for item in fair_order(scores, ["sq-muscle", "sq-protein"])]
    assert ordered[:2] == ["muscle-page", "protein-page"]
    assert "cosmology-paper" not in ordered

    keep_min = SearchRelevanceGate(KeywordEmbeddings(), threshold=0.9, min_keep=1).evaluate(
        subquestions={"sq": "Cosmology muscle?"},
        candidates=[GateCandidate("only", "Protein page", ("sq",))],
    )
    assert keep_min[0].decision == "kept_min"


def _state():
    state = create_initial_workflow_state(job_id="job", research_session_id="session",
                                          question="How to build muscle?", allowed_sources=[ResearchSource.WEB])
    state["research"]["subquestions"] = [SubQuestion(id="sq", question="How to build muscle?", objective="Training")]
    state["research"]["search_queries"] = [SearchQuery(
        id="q-1", subquestion_id="sq", query="muscle growth", purpose="Find", iteration=1,
        providers=["web", "arxiv"],
    )]
    state["research"]["search_results"] = [
        SearchResult(id=result_id, query_id="q-1", provider=provider, canonical_id=f"{provider.value}:{result_id}",
                     title=title, url=f"https://example.com/{result_id}")
        for result_id, provider, title in [
            ("junk", SourceProvider.ARXIV, "Fat convex sets in cosmology"),
            ("good-1", SourceProvider.TAVILY, "Muscle growth meta-analysis"),
            ("good-2", SourceProvider.TAVILY, "Muscle training guide"),
        ]
    ]
    return state


class RecordingExtractor:
    def __init__(self):
        self.calls, self.active, self.max_active = [], 0, 0

    async def extract(self, *, search_result):
        self.calls.append(search_result.id)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0.01)
        self.active -= 1
        return ExtractedSourceContent(url=search_result.url, content="Detailed muscle growth evidence for the report.")


def test_source_processor_rejects_off_topic_candidates_and_carries_deferred_ones():
    async def run():
        state = _state()
        state["research_control"].budget.max_extracted_pages_per_round = 1
        gate = SearchRelevanceGate(KeywordEmbeddings(), threshold=0.7, min_keep=0)
        extractor = RecordingExtractor()
        cleaner = SourceCleaner(min_chars=20, max_chars=1000)

        update = await research_source_processor_node(
            state, source_extractor=extractor, source_cleaner=cleaner, relevance_gate=gate,
        )
        stats = update["research"]["extraction_stats"]
        assert extractor.calls == ["good-1"]
        assert stats["gate_passed_count"] == 2 and stats["gate_rejected_count"] == 1
        # The second relevant page missed the round budget; the off-topic one is dropped.
        assert update["research"]["deferred_search_result_ids"] == ["good-2"]

        # Next round has no new results, but the deferred candidate gets its chance.
        state.update(update)
        state["research"]["iteration_count"] = 1
        again = await research_source_processor_node(
            state, source_extractor=extractor, source_cleaner=cleaner, relevance_gate=gate,
        )
        assert extractor.calls == ["good-1", "good-2"]
        assert again["research"]["extraction_stats"]["carried_over_count"] == 1
        assert again["research"]["deferred_search_result_ids"] == []

    asyncio.run(run())


def test_extraction_runs_concurrently_within_the_limit():
    async def run():
        state = _state()
        state["research"]["search_results"] = [
            result.model_copy(update={"id": f"page-{n}", "canonical_id": f"web:{n}",
                                      "url": f"https://example.com/page-{n}", "provider": SourceProvider.TAVILY})
            for n, result in enumerate(state["research"]["search_results"] * 3)
        ]
        extractor = RecordingExtractor()
        update = await research_source_processor_node(
            state, source_extractor=extractor, source_cleaner=SourceCleaner(min_chars=20, max_chars=1000),
            extraction_concurrency=3,
        )
        assert len(extractor.calls) == 9
        assert 1 < extractor.max_active <= 3
        # Sources keep the ranked selection order despite concurrent extraction.
        assert [source.search_result_ids[0] for source in update["research"]["sources"]] == extractor.calls

    asyncio.run(run())


def test_copies_of_one_work_are_extracted_once_within_and_across_rounds():
    async def run():
        state = _state()
        title = "Muscle growth meta-analysis of resistance training volume"
        state["research"]["search_results"] = [
            SearchResult(id="pubmed", query_id="q-1", provider=SourceProvider.TAVILY, canonical_id="web:pubmed",
                         title=title, url="https://pubmed.ncbi.nlm.nih.gov/1"),
            SearchResult(id="pmc", query_id="q-1", provider=SourceProvider.TAVILY, canonical_id="web:pmc",
                         title=f"{title} - PMC", url="https://pmc.ncbi.nlm.nih.gov/articles/PMC1"),
        ]
        extractor = RecordingExtractor()
        cleaner = SourceCleaner(min_chars=20, max_chars=1000)
        update = await research_source_processor_node(state, source_extractor=extractor, source_cleaner=cleaner)
        assert extractor.calls == ["pmc"]
        assert set(update["research"]["sources"][0].search_result_ids) == {"pubmed", "pmc"}

        # A later round finds the same work on a publisher page: reused, not re-extracted.
        state.update(update)
        state["research"]["iteration_count"] = 1
        state["research"]["search_queries"].append(SearchQuery(
            id="q-2", subquestion_id="sq", query="muscle volume", purpose="Find", iteration=2,
        ))
        state["research"]["search_results"].append(SearchResult(
            id="publisher", query_id="q-2", provider=SourceProvider.TAVILY, canonical_id="web:publisher",
            title=f"{title} | Sports Medicine", url="https://link.example.com/article/1",
        ))
        again = await research_source_processor_node(state, source_extractor=extractor, source_cleaner=cleaner)
        assert extractor.calls == ["pmc"]
        assert again["research"]["extraction_stats"]["reused_count"] == 1
        assert "publisher" in again["research"]["sources"][0].search_result_ids

    asyncio.run(run())


def test_gate_also_scores_against_the_search_phrase_that_found_the_page():
    gate = SearchRelevanceGate(KeywordEmbeddings(), threshold=0.7, min_keep=0)
    candidate = GateCandidate("page", "Protein guide", ("sq",), query_texts=("protein intake",))
    # The abstract subquestion shares no topic with the page; its search phrase does.
    without_phrase = gate.evaluate(subquestions={"sq": "Cosmology?"}, candidates=[GateCandidate("page", "Protein guide", ("sq",))])
    with_phrase = gate.evaluate(subquestions={"sq": "Cosmology?"}, candidates=[candidate])
    assert without_phrase[0].decision == "rejected"
    assert with_phrase[0].decision == "passed" and with_phrase[0].score == 1.0
