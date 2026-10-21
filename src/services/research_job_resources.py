"""Mutable runtime resources owned by one research job."""

from dataclasses import dataclass

from src.retrieval.retrieval_session import RetrievalSession


@dataclass
class ResearchJobResources:
    """Hold non-serializable retrieval resources outside graph state."""

    document_retrieval_session: RetrievalSession | None = None
    # Web sources are indexed incrementally into one session reused across rounds.
    web_retrieval_session: RetrievalSession | None = None

    def clear(self) -> None:
        """Release every job-scoped retrieval resource."""

        sessions = [self.document_retrieval_session, self.web_retrieval_session]
        self.document_retrieval_session = None
        self.web_retrieval_session = None
        errors = []
        for session in sessions:
            if session is None:
                continue
            # Clear every session even if an earlier one fails.
            try:
                session.clear()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
