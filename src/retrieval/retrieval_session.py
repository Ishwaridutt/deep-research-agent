from itertools import zip_longest

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
        print("Clearing before building a new retrieval session...")
        self.clear()

        if not documents:
            return

        # Chunk the corpus once and build both lexical and semantic retrieval indexes.
        self.chunks = self.chunker.chunk_documents(documents)
        self.hybrid_retriever.build(self.chunks)
        print('\nRetrieval session chunking and loading completed...')

    def add(self, documents: list[RetrievalDocument]) -> int:
        """
        Chunks and indexes only documents not yet in this session.

        A job-owned session grows across research rounds, so earlier sources
        are never re-chunked or re-embedded. Returns the number of new documents.
        """
        indexed_ids = {chunk.document_id for chunk in self.chunks}
        new_documents = [
            document for document in dict((item.document_id, item) for item in documents).values()
            if document.document_id not in indexed_ids
        ]
        print(f"\nRetrieval session add: {len(new_documents)} new documents; {len(documents) - len(new_documents)} already indexed")
        if not new_documents:
            return 0
        new_chunks = self.chunker.chunk_documents(new_documents)
        self.chunks.extend(new_chunks)
        self.hybrid_retriever.add(new_chunks)
        print('\nRetrieval session incremental indexing completed...')
        return len(new_documents)

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Retrieves and builds bounded context for one research subquestion."""

        retrieval_query = self._build_retrieval_query(request)
        print(f"\nstarting retrieval for retrieval_query: ", retrieval_query)

        # Search each intent separately, then interleave rankings so a broad query
        # cannot hide the narrow follow-up searches before context selection.
        # The subquestion always leads: it states the evidence need in natural
        # language, while web queries are tuned for search engines.
        queries = list(dict.fromkeys(
            query.strip()
            for query in [request.subquestion, *request.search_queries]
            if query and query.strip()
        ))
        print("\nretrieval search intents:", queries)
        rankings = [self.search(query) for query in queries]
        retrieved_chunks = []
        seen_chunk_ids: set[str] = set()
        for rank_group in zip_longest(*rankings):
            for result in rank_group:
                if result is not None and result.chunk.chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(result.chunk.chunk_id)
                    retrieved_chunks.append(RetrievedChunk(result.chunk, len(retrieved_chunks) + 1))
        print(f"\ncompleted retrieval for retrieval_query: ", retrieval_query, ' retrieved_chunks: ',len(retrieved_chunks))

        # Convert the ranked results into token-bounded context for the evidence extractor.
        result = self.context_builder.build(
            request=request,
            retrieval_query=retrieval_query,
            retrieved_chunks=retrieved_chunks,
        )
        selected_ids = {item.chunk.chunk_id for item in result.chunks}
        result.query_matches = [
            {
                "query": query,
                "retrieved_count": len(ranking),
                "selected_count": sum(item.chunk.chunk_id in selected_ids for item in ranking),
                "selected_chunk_ids": [item.chunk.chunk_id for item in ranking if item.chunk.chunk_id in selected_ids],
            }
            for query, ranking in zip(queries, rankings)
        ]
        print("\nretrieval query matches:", result.query_matches)
        return result

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Runs raw hybrid retrieval without building LLM context."""

        return self.hybrid_retriever.search(query, top_k=top_k or self.top_k)

    def clear(self) -> None:
        """Releases all research-session-specific retrieval data."""

        self.hybrid_retriever.clear()
        self.chunks = []

    def _build_retrieval_query(self, request: RetrievalRequest) -> str:
        """Describe the request in result metadata; actual searches stay separate."""

        query_parts = [request.subquestion.strip()]

        for search_query in request.search_queries:
            search_query = search_query.strip()

            if search_query and search_query not in query_parts:
                query_parts.append(search_query)

        return "\n".join(query_parts)
