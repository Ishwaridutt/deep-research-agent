from collections import defaultdict
from collections.abc import Callable

from src.models.retrieval_model import RetrievalRequest, RetrievalResult, RetrievedChunk


class RetrievalContextBuilder:
    """Builds token-bounded LLM context from hybrid-ranked retrieval results."""

    def __init__(
        self,
        token_counter: Callable[[str], int],
        max_context_tokens: int,
        max_chunks_per_document: int = 3,
    ):
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be greater than 0.")

        if max_chunks_per_document <= 0:
            raise ValueError("max_chunks_per_document must be greater than 0.")

        self.token_counter = token_counter
        self.max_context_tokens = max_context_tokens
        self.max_chunks_per_document = max_chunks_per_document

    def build(
        self,
        request: RetrievalRequest,
        retrieval_query: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> RetrievalResult:
        """
        Selects diverse chunks and builds the final bounded retrieval context.
        """

        selected_chunks: list[RetrievedChunk] = []
        context_parts: list[str] = []
        document_counts: dict[str, int] = defaultdict(int)
        seen_chunk_ids: set[str] = set()

        used_tokens = 0
        separator_tokens = self.token_counter("\n\n")

        # Walk through the fused ranking while preventing duplicate chunks and source domination.
        for retrieved_chunk in retrieved_chunks:
            chunk = retrieved_chunk.chunk
            # remove duplicate chunk IDs
            if chunk.chunk_id in seen_chunk_ids:
                continue
            # max {num} chunks / source
            if document_counts[chunk.document_id] >= self.max_chunks_per_document:
                continue

            formatted_chunk = self._format_chunk(len(selected_chunks) + 1, retrieved_chunk)
            # Count only the new part instead of re-tokenizing the whole context each time.
            chunk_tokens = self.token_counter(formatted_chunk)
            candidate_tokens = used_tokens + chunk_tokens + (separator_tokens if context_parts else 0)

            # Skip chunks that would push the final retrieval context beyond the configured budget.
            if candidate_tokens > self.max_context_tokens:
                continue

            context_parts.append(formatted_chunk)
            used_tokens = candidate_tokens
            selected_chunks.append(retrieved_chunk)
            seen_chunk_ids.add(chunk.chunk_id)
            document_counts[chunk.document_id] += 1

        context = "\n\n".join(context_parts)

        return RetrievalResult(
            subquestion_id=request.subquestion_id,
            retrieval_query=retrieval_query,
            chunks=selected_chunks,
            context=context,
            estimated_tokens=self.token_counter(context) if context else 0,
        )

    def _format_chunk(self, evidence_number: int, retrieved_chunk: RetrievedChunk) -> str:
        """Formats one retrieved chunk while retaining useful provenance for evidence extraction."""

        chunk = retrieved_chunk.chunk

        parts = [
            f"[Evidence {evidence_number}]",
            f"Chunk ID: {chunk.chunk_id}",
            f"Document ID: {chunk.document_id}",
        ]

        if chunk.title:
            parts.append(f"Title: {chunk.title}")

        if chunk.section:
            parts.append(f"Section: {chunk.section}")

        if chunk.metadata.get("url"):
            parts.append(f"URL: {chunk.metadata['url']}")

        parts.append(f"Content:\n{chunk.content}")

        return "\n".join(parts)

    