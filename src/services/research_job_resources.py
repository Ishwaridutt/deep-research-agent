"""Mutable runtime resources owned by one research job."""

from dataclasses import dataclass

from src.retrieval.retrieval_session import RetrievalSession


@dataclass
class ResearchJobResources:
    """Hold non-serializable retrieval resources outside graph state."""

    document_retrieval_session: RetrievalSession | None = None

    def clear(self) -> None:
        """Release every job-scoped retrieval resource."""

        if self.document_retrieval_session is None:
            return

        try:
            self.document_retrieval_session.clear()
        finally:
            self.document_retrieval_session = None
