from typing import Any
from chonkie import RecursiveChunker

from src.models.retrieval_model import RetrievalChunk, RetrievalDocument


class RetrievalChunker:
    """Chunks documents using Chonkie's recursive structure-aware chunker."""

    def __init__(self, tokenizer: Any, chunk_size: int = 400):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        # Chonkie owns the paragraph -> sentence -> smaller-boundary chunking logic.
        # Pass the actual tokenizer so Chonkie can both count and encode tokens
        # when recursive structural splitting reaches its token-level fallback.
        self.chunker = RecursiveChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
        )

    def chunk_documents(self, documents: list[RetrievalDocument]) -> list[RetrievalChunk]:
        """Chunks multiple retrieval documents."""

        chunks: list[RetrievalChunk] = []

        # Process each source independently so no chunk can span multiple documents.
        for document in documents:
            chunks.extend(self.chunk_document(document))

        print('Chunking completed. Total chunks:', len(chunks))
        if chunks:
            print(
                "Maximum chunk token count:",
                max(chunk.token_count for chunk in chunks),
            )
        return chunks

    def chunk_document(self, document: RetrievalDocument) -> list[RetrievalChunk]:
        """Chunks a single retrieval document while preserving provenance."""

        if not document.content or not document.content.strip():
            return []

        generated_chunks = self.chunker(document.content)
        chunks: list[RetrievalChunk] = []

        # Convert Chonkie's chunk objects into our retrieval-domain representation.
        for chunk_index, generated_chunk in enumerate(generated_chunks):
            metadata = document.metadata.copy()
            metadata["start_index"] = generated_chunk.start_index
            metadata["end_index"] = generated_chunk.end_index

            chunks.append(
                RetrievalChunk(
                    chunk_id=f"{document.document_id}:{chunk_index}",
                    document_id=document.document_id,
                    content=generated_chunk.text,
                    source_type=document.source_type,
                    chunk_index=chunk_index,
                    token_count=generated_chunk.token_count,
                    title=document.title,
                    section=None,
                    metadata=metadata,
                )
            )

        return chunks

