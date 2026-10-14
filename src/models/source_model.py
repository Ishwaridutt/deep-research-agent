"""Models representing discovered and extracted research sources."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Identifies where a normalized research source originated."""

    WEB = "web"

    DOCUMENT = "document"

    # Future phases.
    MCP = "mcp"
    API = "api"
    IMAGE = "image"


class SourceProvider(str, Enum):
    """Identifies the adapter that supplied normalized source content."""

    TAVILY = "tavily"
    BRAVE = "brave"
    ARXIV = "arxiv"
    WIKIPEDIA = "wikipedia"
    PRIVATE_DOCUMENT = "private_document"


class SearchResult(BaseModel):
    """Normalized search result returned by a search provider."""

    id: str = Field(
        min_length=1,
    )

    query_id: str = Field(
        min_length=1,
    )

    # Identifies which adapter discovered the result for provenance and fallback tracking.
    provider: SourceProvider

    # Stable source identity used to merge the same result returned by multiple providers.
    canonical_id: str = Field(min_length=1)

    title: str = Field(
        min_length=1,
    )

    url: str = Field(
        min_length=1,
    )

    snippet: str | None = None

    provider_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class Source(BaseModel):
    """Normalized source content used by evidence extraction."""

    id: str = Field(
        min_length=1,
    )

    source_type: SourceType

    # Identifies which adapter supplied the normalized content.
    provider: SourceProvider

    # Represents the underlying source independently of application-generated IDs.
    canonical_id: str = Field(min_length=1)

    search_result_ids: list[str] = Field(
        default_factory=list,
    )

    title: str = Field(
        min_length=1,
    )

    uri: str = Field(
        min_length=1,
    )

    content: str = Field(
        min_length=1,
    )

    retrieved_at: datetime

    provider_metadata: dict[str, Any] = Field(default_factory=dict)


