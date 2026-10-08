from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from src.models.retrieval_model import RetrievalChunk, RetrievedChunk
from src.retrieval.retrievers.retrieval_bm25_retriever import RetrievalBM25Retriever


class RetrievalHybridRetriever:
    """Performs semantic and BM25 retrieval followed by weighted RRF."""

    def __init__(
        self,
        embeddings: Embeddings,
        candidate_k: int = 10,
        semantic_weight: float = 0.5,
        lexical_weight: float = 0.5,
        rrf_c: int = 60,
    ):
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than 0.")

        if semantic_weight < 0 or lexical_weight < 0 or semantic_weight + lexical_weight <= 0:
            raise ValueError("retrieval weights must contain at least one positive value.")

        if rrf_c <= 0:
            raise ValueError("rrf_c must be greater than 0.")

        # Normalize weights so their meaning remains intuitive even if configuration values do not sum to one.
        total_weight = semantic_weight + lexical_weight

        self.embeddings = embeddings
        self.candidate_k = candidate_k
        self.semantic_weight = semantic_weight / total_weight
        self.lexical_weight = lexical_weight / total_weight
        self.rrf_c = rrf_c

        self.vector_store: InMemoryVectorStore | None = None
        self.retriever: EnsembleRetriever | None = None
        self.chunk_lookup: dict[str, RetrievalChunk] = {}

    def build(self, chunks: list[RetrievalChunk]) -> None:
        """Builds the session-scoped semantic and BM25 retrievers."""

        self.clear()

        if not chunks:
            return

        # Convert domain chunks once so both lexical and semantic retrieval operate over the exact same corpus.
        documents = [self._to_langchain_document(chunk) for chunk in chunks]
        self.chunk_lookup = {chunk.chunk_id: chunk for chunk in chunks}
        result_count = min(self.candidate_k, len(documents))

        # Build the in-memory dense index using the shared application-scoped embedding model.
        self.vector_store = InMemoryVectorStore(embedding=self.embeddings)
        self.vector_store.add_documents(documents=documents)
        semantic_retriever = self.vector_store.as_retriever(search_kwargs={"k": result_count})

        # Build the sparse BM25 index from the same searchable document representation.
        bm25_retriever = RetrievalBM25Retriever.from_documents(documents, k=result_count)

        # LangChain handles weighted Reciprocal Rank Fusion and chunk-level deduplication.
        self.retriever = EnsembleRetriever(
            retrievers=[semantic_retriever, bm25_retriever],
            weights=[self.semantic_weight, self.lexical_weight],
            c=self.rrf_c,
            id_key="chunk_id",
        )

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Returns hybrid-ranked chunks for a query."""

        if not query or not query.strip():
            raise ValueError("query cannot be empty.")

        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        if self.retriever is None:
            return []

        # EnsembleRetriever executes semantic and BM25 retrieval and returns the final RRF-ordered documents.
        documents = self.retriever.invoke(query.strip())
        result_count = min(top_k or self.candidate_k, len(documents))
        retrieved_chunks: list[RetrievedChunk] = []

        # Restore our domain objects so LangChain types do not leak into the rest of the application.
        for rank, document in enumerate(documents[:result_count], start=1):
            chunk_id = document.metadata["chunk_id"]
            chunk = self.chunk_lookup.get(chunk_id)

            if chunk is not None:
                retrieved_chunks.append(RetrievedChunk(chunk=chunk, rank=rank))

        return retrieved_chunks

    def clear(self) -> None:
        """Drops all research-session-specific indexes and chunk references."""

        self.retriever = None
        self.vector_store = None
        self.chunk_lookup = {}

    def _to_langchain_document(self, chunk: RetrievalChunk) -> Document:
        """Converts a retrieval chunk into the searchable LangChain representation."""

        # Add title and section to the searchable text because they often contain useful lexical and semantic signals.
        searchable_parts = [part for part in [chunk.title, chunk.section, chunk.content] if part]
        searchable_content = "\n\n".join(searchable_parts)

        metadata = chunk.metadata.copy()
        metadata.update(
            {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source_type": chunk.source_type,
                "chunk_index": chunk.chunk_index,
                "title": chunk.title,
                "section": chunk.section,
            }
        )

        return Document(id=chunk.chunk_id, page_content=searchable_content, metadata=metadata)

    