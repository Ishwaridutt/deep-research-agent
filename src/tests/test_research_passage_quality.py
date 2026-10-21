"""Offline regressions for noisy retrieval, job corpus reuse, and prompt scope."""

from langchain_core.embeddings import Embeddings

from src.agents.research.research_retrieval_node import _web_documents
from src.graph.graph_state import create_initial_workflow_state
from src.models import ResearchSource, SourceType, SubQuestion
from src.models.retrieval_model import RetrievalChunk
from src.prompts.query_generator_prompt import build_query_generator_messages
from src.prompts.research_planner_prompt import build_planner_messages
from src.retrieval.retrievers.retrieval_content_quality import is_public_noise
from src.retrieval.retrievers.retrieval_hybrid_retriever import RetrievalHybridRetriever


def chunk(text, identifier="chunk", source_type="web"):
    return RetrievalChunk(identifier, "source", text, source_type, 0, 100)


def test_reference_lists_and_navigation_do_not_take_hybrid_slots():
    references = chunk("## References\n1. Smith, A. Training adaptations. Journal of Science 2020; 12:1.\n"
                       "2. Brown, B. Hypertrophy research. Sports Journal 2021; 13:4.\n"
                       "3. Jones, C. Resistance training. Medical Journal 2022; 14:5.", "refs")
    navigation = chunk("[Download](https://example.org/pdf)\nShare\nCite", "nav")
    useful = chunk("Resistance training increased muscle thickness by 12% in the trial [1].", "finding")

    class FakeEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [[1.0, 0.5] for _ in texts]

        def embed_query(self, text):
            return [1.0, 0.5]

    retriever = RetrievalHybridRetriever(FakeEmbeddings(), candidate_k=3)
    retriever.build([references, navigation, useful])
    assert set(retriever.chunk_lookup) == {"finding"}
    assert [item.chunk.chunk_id for item in retriever.search("training", top_k=1)] == ["finding"]


def test_noise_filter_preserves_numbered_facts_tables_citations_and_private_text():
    passages = [
        "1. Train consistently.\n2. Progress gradually.\n3. Monitor fatigue.",
        "## References\nThe findings differ from Smith et al. (2020), who reported 12% growth [3].",
        "1. Smith A. (2020) reported a 12% increase in the trial; doi:10.1000/example.",
        "| Group | Result |\n|---|---:|\n| A | 12% [1] |\n| B | 9% [2] |",
    ]
    assert not any(is_public_noise(chunk(text)) for text in passages)
    assert not is_public_noise(chunk("Share\nCite", source_type="document"))


def test_single_reference_and_undotted_author_initials_are_detected():
    assert is_public_noise(chunk(
        "134. Schoenfeld BJ, Grgic J, Krieger J. Resistance training frequency. J Sports Sci. 2019; 37 (11): 1286–95. PMID:30558493 doi:10.1080/example"
    ))
    assert is_public_noise(chunk(
        "7. Kraemer, W.J.; Ratamess, N.A. Progression and exercise prescription. Med. Sci. Sports Exerc. **2004**, 36, 674–688. [[CrossRef](https://doi.org/10.1249/example)]"
    ))


def test_configurable_extraction_allowance_keeps_the_job_wide_cap():
    from types import SimpleNamespace
    from src.config.settings import ResearchLimitSettings
    from src.services.research_service import ResearchService
    limits = ResearchLimitSettings(_env_file=None, max_extracted_pages_per_round=7)
    service = ResearchService(graph=None, job_store=None, settings=SimpleNamespace(research_limits=limits))
    state = service._create_state(question="Topic", source_selection=None, owner_id=None, research_session_id=None)
    assert state["research_control"].budget.max_extracted_pages_per_round == 7
    assert state["research_control"].budget.max_extracted_pages == 60
    assert ResearchLimitSettings(_env_file=None).max_extracted_pages_per_round == 30


def test_followup_public_corpus_includes_earlier_job_sources_but_not_private_sources():
    from datetime import UTC, datetime
    from src.models import Source, SourceProvider
    state = create_initial_workflow_state(job_id="job", research_session_id="session",
                                          question="Topic", allowed_sources=[ResearchSource.WEB])
    state["research"]["sources"] = [Source(
        id="old-source", source_type=SourceType.WEB, provider=SourceProvider.ARXIV,
        canonical_id="arxiv:old", title="Earlier source", uri="https://arxiv.org/abs/2410.16169",
        content="Earlier job content", search_result_ids=["old-result"], retrieved_at=datetime.now(UTC),
    )]
    state["research"]["iteration_count"] = 2
    assert [document.document_id for document in _web_documents(state)] == ["old-source"]
    state["research"]["sources"][0].source_type = SourceType.DOCUMENT
    assert _web_documents(state) == []


def test_prompts_prioritize_explicit_facets_and_followup_needs_without_topic_hardcoding():
    planner = build_planner_messages(question="Compare database performance and operating cost",
                                     max_subquestions=5, current_date="2026-10-20")[0].content
    task = SubQuestion(id="followup", parent_subquestion_id="root", question="Unresolved costs?",
                       objective="Find operating and migration costs")
    messages = build_query_generator_messages(original_question="Compare databases", subquestions=[task],
                                              max_queries=3, max_query_words=12, previous_queries=["earlier search"],
                                              current_date="2026-10-20")
    assert "independent part explicitly requested" in planner
    assert "different sources" in planner
    assert "themes the user did not ask for" in planner
    assert "specific facts, numbers, or recommendations" in messages[0].content
    assert "at most 12 words" in messages[0].content
    assert "follow-up on unresolved gaps" in messages[1].content
    assert '"subquestion_id": "followup"' in messages[1].content
    assert "muscle" not in planner.lower() + messages[0].content.lower()
