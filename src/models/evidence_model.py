"""Evidence models used by the research pipeline."""

from pydantic import BaseModel, Field


class EvidenceCandidate(BaseModel):
    """Evidence extracted semantically by the LLM from retrieved context."""

    chunk_id: str = Field(
        min_length=1,
    )

    content: str = Field(
        min_length=1,
    )

    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
    )


class EvidenceExtractionOutput(BaseModel):
    """Structured output returned by the evidence extractor."""

    evidence: list[EvidenceCandidate]


class Evidence(BaseModel):
    """Accepted evidence with application-managed provenance."""

    id: str = Field(
        min_length=1,
    )

    subquestion_id: str = Field(
        min_length=1,
    )

    source_id: str = Field(
        min_length=1,
    )

    content: str = Field(
        min_length=1,
    )

    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    