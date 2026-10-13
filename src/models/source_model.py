"""Models representing discovered and extracted research sources."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    """Identifies where a normalized research source originated."""

    WEB = "web"

    DOCUMENT = "document"

    # Future phases.
    MCP = "mcp"
    API = "api"
    IMAGE = "image"


class SearchResult(BaseModel):
    """Normalized search result returned by a search provider."""

    id: str = Field(
        min_length=1,
    )

    query_id: str = Field(
        min_length=1,
    )

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


class Source(BaseModel):
    """Normalized source content used by evidence extraction."""

    id: str = Field(
        min_length=1,
    )

    source_type: SourceType

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


