from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalDocument:
    """Canonical document representation accepted by the retrieval subsystem."""

    document_id: str
    content: str
    source_type: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalChunk:
    """A chunk generated from a retrieval document while preserving source provenance."""

    chunk_id: str
    document_id: str
    content: str
    source_type: str
    chunk_index: int
    token_count: int
    title: str | None = None
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalRequest:
    """Represents one subquestion-level retrieval request."""

    subquestion_id: str
    research_question: str
    subquestion: str
    search_queries: list[str] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    """A retrieval chunk ordered by the final hybrid retrieval ranking."""

    chunk: RetrievalChunk
    rank: int


@dataclass
class RetrievalResult:
    """Final retrieval output consumed by the evidence extraction stage."""

    subquestion_id: str
    retrieval_query: str
    chunks: list[RetrievedChunk]
    context: str
    estimated_tokens: int

