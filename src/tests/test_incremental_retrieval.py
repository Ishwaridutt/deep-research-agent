"""Offline tests for incremental indexing in the shared hybrid retriever and session."""

from src.models.retrieval_model import RetrievalChunk, RetrievalDocument
from src.retrieval.context.retrieval_context_builder import RetrievalContextBuilder
from src.retrieval.retrieval_session import RetrievalSession
from src.retrieval.retrievers.retrieval_hybrid_retriever import RetrievalHybridRetriever


class CountingEmbeddings:
    """Fake embeddings that record how many texts were embedded."""

    def __init__(self):
        self.embedded = []

    def _vector(self, text):
        return [float("protein" in text), float("training" in text), 1.0]

    def embed_documents(self, texts):
        self.embedded.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        return self._vector(text)


def _chunk(identifier, content):
    return RetrievalChunk(
        chunk_id=f"{identifier}:0", document_id=identifier, content=content,
        source_type="web", chunk_index=0, token_count=5, title=identifier,
    )


def test_hybrid_retriever_add_embeds_only_new_chunks_and_searches_whole_corpus():
    embeddings = CountingEmbeddings()
    retriever = RetrievalHybridRetriever(embeddings, candidate_k=5)
    retriever.add([_chunk("first", "Daily protein intake recommendations for adults.")])
    retriever.add([
        _chunk("first", "Daily protein intake recommendations for adults."),
        _chunk("second", "Resistance training volume and frequency guidance."),
    ])

    # The repeated chunk was not embedded again.
    assert len(embeddings.embedded) == 2
    found = {item.chunk.document_id for item in retriever.search("protein training guidance")}
    assert found == {"first", "second"}

    retriever.clear()
    assert retriever.search("protein") == [] and retriever.documents == []


def test_session_add_chunks_only_documents_not_yet_indexed():
    class Chunker:
        def __init__(self):
            self.calls = []

        def chunk_documents(self, documents):
            self.calls.append([document.document_id for document in documents])
            return [_chunk(document.document_id, document.content) for document in documents]

    chunker = Chunker()
    session = RetrievalSession(
        chunker=chunker,
        hybrid_retriever=RetrievalHybridRetriever(CountingEmbeddings(), candidate_k=5),
        context_builder=RetrievalContextBuilder(len, 10_000),
    )
    first = RetrievalDocument(document_id="a", content="Protein intake guidance for adults.", source_type="web")
    second = RetrievalDocument(document_id="b", content="Training frequency guidance for adults.", source_type="web")

    assert session.add([first]) == 1
    assert session.add([first, second]) == 1
    assert session.add([first, second]) == 0
    assert chunker.calls == [["a"], ["b"]]
    assert {chunk.document_id for chunk in session.chunks} == {"a", "b"}
