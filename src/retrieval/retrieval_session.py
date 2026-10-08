from src.models.retrieval_model import RetrievalChunk, RetrievalDocument, RetrievalRequest, RetrievalResult, RetrievedChunk
from src.retrieval.chunking.retrieval_chunker import RetrievalChunker
from src.retrieval.context.retrieval_context_builder import RetrievalContextBuilder
from src.retrieval.retrievers.retrieval_hybrid_retriever import RetrievalHybridRetriever


class RetrievalSession:
    """Owns temporary chunks and retrieval indexes for one research session."""

    def __init__(
        self,
        chunker: RetrievalChunker,
        hybrid_retriever: RetrievalHybridRetriever,
        context_builder: RetrievalContextBuilder,
        top_k: int = 10,
    ):
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        self.chunker = chunker
        self.hybrid_retriever = hybrid_retriever
        self.context_builder = context_builder
        self.top_k = top_k
        self.chunks: list[RetrievalChunk] = []

    def build(self, documents: list[RetrievalDocument]) -> None:
        """Chunks and indexes all documents available to the research session."""

        self.clear()

        if not documents:
            return

        # Chunk the corpus once and build both lexical and semantic retrieval indexes.
        self.chunks = self.chunker.chunk_documents(documents)
        self.hybrid_retriever.build(self.chunks)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Retrieves and builds bounded context for one research subquestion."""

        retrieval_query = self._build_retrieval_query(request)

        # Search the shared session corpus using BM25 + semantic retrieval + RRF.
        retrieved_chunks = self.hybrid_retriever.search(retrieval_query, top_k=self.top_k)

        # Convert the ranked results into token-bounded context for the evidence extractor.
        return self.context_builder.build(
            request=request,
            retrieval_query=retrieval_query,
            retrieved_chunks=retrieved_chunks,
        )

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Runs raw hybrid retrieval without building LLM context."""

        return self.hybrid_retriever.search(query, top_k=top_k or self.top_k)

    def clear(self) -> None:
        """Releases all research-session-specific retrieval data."""

        self.hybrid_retriever.clear()
        self.chunks = []

    def _build_retrieval_query(self, request: RetrievalRequest) -> str:
        """Builds the search query used by both lexical and semantic retrieval."""

        query_parts = [request.subquestion.strip()]

        # Search queries often contain useful keywords and named entities that improve BM25 recall.
        for search_query in request.search_queries:
            search_query = search_query.strip()

            if search_query and search_query not in query_parts:
                query_parts.append(search_query)

        return "\n".join(query_parts)

    