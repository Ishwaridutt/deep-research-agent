from collections.abc import Callable

from src.retrieval.chunking.retrieval_chunker import RetrievalChunker
from src.retrieval.context.retrieval_context_builder import RetrievalContextBuilder
from src.retrieval.embeddings.retrieval_embedding_service import RetrievalEmbeddingService
from src.retrieval.retrieval_session import RetrievalSession
from src.retrieval.retrievers.retrieval_hybrid_retriever import RetrievalHybridRetriever


class RetrievalSessionFactory:
    """Creates isolated retrieval sessions while reusing the shared embedding model."""

    def __init__(
        self,
        embedding_service: RetrievalEmbeddingService,
        context_token_counter: Callable[[str], int],
        chunk_size: int = 400,
        candidate_k: int = 10,
        top_k: int = 5,
        semantic_weight: float = 0.5,
        lexical_weight: float = 0.5,
        rrf_c: int = 60,
        max_context_tokens: int = 3000,
        max_chunks_per_document: int = 3,
    ):
        self.embedding_service = embedding_service
        self.context_token_counter = context_token_counter
        self.chunk_size = chunk_size
        self.candidate_k = candidate_k
        self.top_k = top_k
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.rrf_c = rrf_c
        self.max_context_tokens = max_context_tokens
        self.max_chunks_per_document = max_chunks_per_document

    def create_session(self) -> RetrievalSession:
        """Creates a fresh retrieval session for one research job."""

        # Chonkie needs the actual tokenizer because recursive splitting may
        # eventually fall back to token-level splitting.
        chunker = RetrievalChunker(
            tokenizer=self.embedding_service.tokenizer,
            chunk_size=self.chunk_size,
        )

        hybrid_retriever = RetrievalHybridRetriever(
            embeddings=self.embedding_service.embeddings,
            candidate_k=self.candidate_k,
            semantic_weight=self.semantic_weight,
            lexical_weight=self.lexical_weight,
            rrf_c=self.rrf_c,
        )

        # Context construction only needs a callable that returns a token count.
        context_builder = RetrievalContextBuilder(
            token_counter=self.context_token_counter,
            max_context_tokens=self.max_context_tokens,
            max_chunks_per_document=self.max_chunks_per_document,
        )

        return RetrievalSession(
            chunker=chunker,
            hybrid_retriever=hybrid_retriever,
            context_builder=context_builder,
            top_k=self.top_k,
        )

    