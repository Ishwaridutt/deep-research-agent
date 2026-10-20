"""Evidence models used by the research pipeline."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.models.source_model import SourceType


class EvidenceCandidate(BaseModel):
    """A chunk relevance judgment; source text stays application-owned."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(
        min_length=1,
    )

    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
    )


class EvidenceExtractionOutput(BaseModel):
    """Chunk scores, retaining the existing schema name and LLM role."""

    evidence: list[EvidenceCandidate] = Field(description="Exactly one relevance score per supplied chunk ID.")

    @field_validator("evidence", mode="before")
    @classmethod
    def unwrap_items_container(cls, value: object) -> object:
        """Accept the schema-keyword wrapper emitted by some providers."""

        if isinstance(value, dict) and set(value) == {"items"}:
            return value["items"]
        return value


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

    chunk_id: str = Field(min_length=1)

    source_type: SourceType

    provenance: dict[str, Any] = Field(default_factory=dict)

    content: str = Field(
        min_length=1,
    )

    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
    )
