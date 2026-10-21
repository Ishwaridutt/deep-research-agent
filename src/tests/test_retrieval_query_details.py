"""Per-intent observability through the existing, shared retrieval path."""

from src.models.retrieval_model import RetrievalChunk, RetrievalRequest, RetrievedChunk
from src.retrieval.context.retrieval_context_builder import RetrievalContextBuilder
from src.retrieval.retrieval_session import RetrievalSession


def test_query_matches_track_overlap_deduplication_and_context_limits():
    def chunk(identifier):
        return RetrievedChunk(RetrievalChunk(identifier, "same-document", identifier, "web", 0, 1), 1)

    a, b, c = [chunk(identifier) for identifier in ("a", "b", "c")]

    class Retriever:
        def __init__(self):
            self.calls = []

        def search(self, query, *, top_k):
            self.calls.append(query)
            return {"Question?": [], "first": [a, b], "second": [a, c]}[query]

    retriever = Retriever()
    session = RetrievalSession(None, retriever, RetrievalContextBuilder(len, 10000, max_chunks_per_document=2))
    result = session.retrieve(RetrievalRequest("task", "Topic?", "Question?", [" first ", "first", "second"]))
    # The subquestion is always searched first as its own intent.
    assert retriever.calls == ["Question?", "first", "second"]
    assert [item.chunk.chunk_id for item in result.chunks] == ["a", "b"]
    assert result.query_matches == [
        {"query": "Question?", "retrieved_count": 0, "selected_count": 0, "selected_chunk_ids": []},
        {"query": "first", "retrieved_count": 2, "selected_count": 2, "selected_chunk_ids": ["a", "b"]},
        {"query": "second", "retrieved_count": 2, "selected_count": 1, "selected_chunk_ids": ["a"]},
    ]


def test_no_queries_falls_back_to_subquestion_even_without_matches():
    class Retriever:
        def search(self, query, *, top_k):
            assert query == "Question?"
            return []

    session = RetrievalSession(None, Retriever(), RetrievalContextBuilder(len, 100))
    result = session.retrieve(RetrievalRequest("task", "Topic?", " Question? ", [" "]))
    assert result.query_matches == [{"query": "Question?", "retrieved_count": 0, "selected_count": 0, "selected_chunk_ids": []}]
    assert result.chunks == []
