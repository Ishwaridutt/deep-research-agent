"""Data models for private document ingestion and retrieval."""

from datetime import datetime, timezone
from enum import Enum
from typing import Self
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc)


def generate_id() -> str:
    """Generate a unique identifier."""
    return str(uuid4())


class DocumentFormat(str, Enum):
    """Supported private document formats."""

    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "md"


class DocumentStatus(str, Enum):
    """High-level lifecycle state of an uploaded document."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class DocumentProcessingStage(str, Enum):
    """Individual stages of the document-processing pipeline."""

    UPLOADED = "uploaded"
    VALIDATING = "validating"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchSourceMode(str, Enum):
    """Sources that may be used for a research request."""

    WEB = "web"
    DOCUMENTS = "documents"
    HYBRID = "hybrid"


class DocumentRecord(BaseModel):
    """
    Metadata for an uploaded private document.

    The original document is stored on disk. The storage key points
    to that file without exposing its physical path to API clients.
    """

    document_id: str = Field(default_factory=generate_id)
    owner_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    document_format: DocumentFormat
    content_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    content_hash: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[a-f0-9]{64}$",
    )
    # Internal reference into the configured upload directory.
    # Example:
    #   9bf31c6a-...-993.pdf
    storage_key: str = Field(
        min_length=1,
        exclude=True,
    )
    status: DocumentStatus = DocumentStatus.UPLOADED
    chunk_count: int = Field(
        default=0,
        ge=0,
    )
    error_message: str | None = None
    created_at: datetime = Field(
        default_factory=utc_now,
    )
    updated_at: datetime = Field(
        default_factory=utc_now,
    )


class DocumentSourceLocation(BaseModel):
    """
    Location of a chunk within the original document.

    This metadata will later be used for document citations.
    """

    page_start: int | None = Field(
        default=None,
        ge=1,
    )

    page_end: int | None = Field(
        default=None,
        ge=1,
    )

    section_title: str | None = None

    char_start: int | None = Field(
        default=None,
        ge=0,
    )

    char_end: int | None = Field(
        default=None,
        ge=0,
    )

    @model_validator(mode="after")
    def validate_ranges(self) -> Self:
        if self.page_end is not None:
            if self.page_start is None:
                raise ValueError(
                    "page_start is required when page_end is provided."
                )

            if self.page_end < self.page_start:
                raise ValueError(
                    "page_end cannot precede page_start."
                )

        if self.char_end is not None:
            if self.char_start is None:
                raise ValueError(
                    "char_start is required when char_end is provided."
                )

            if self.char_end < self.char_start:
                raise ValueError(
                    "char_end cannot precede char_start."
                )

        return self


class StoredDocumentFile(BaseModel):
    """Result of storing an uploaded document on disk."""

    storage_key: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    content_hash: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[a-f0-9]{64}$",
    )


class DocumentCollection(BaseModel):
    """
    Reusable collection of private documents.
    """

    collection_id: str = Field(default_factory=generate_id)
    owner_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    document_ids: list[str] = Field(
        default_factory=list,
    )
    created_at: datetime = Field(
        default_factory=utc_now,
    )


class DocumentProcessingEvent(BaseModel):
    """
    Progress event generated while processing a private document.

    These events will later drive the document-processing UI.
    """

    event_id: str = Field(default_factory=generate_id)
    document_id: str = Field(min_length=1)
    owner_id: str = Field(min_length=1)
    stage: DocumentProcessingStage
    progress_percent: int | None = Field(
        default=None,
        ge=0,
        le=100,
    )
    message: str | None = None
    created_at: datetime = Field(
        default_factory=utc_now,
    )


class ResearchSourceSelection(BaseModel):
    """
    Defines which source types a research request may use.

    Authorization of selected document IDs must happen separately
    before retrieval.
    """

    mode: ResearchSourceMode = ResearchSourceMode.WEB
    document_ids: list[str] = Field(
        default_factory=list,
    )
    collection_ids: list[str] = Field(
        default_factory=list,
    )
    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        uses_documents = self.mode in {
            ResearchSourceMode.DOCUMENTS,
            ResearchSourceMode.HYBRID,
        }

        has_document_selection = (
            bool(self.document_ids)
            or bool(self.collection_ids)
        )

        if uses_documents and not has_document_selection:
            raise ValueError(
                "Document or hybrid research requires at least "
                "one document or collection."
            )

        return self


class ParsedDocumentBlock(BaseModel):
    """Extracted logical block from a document."""

    text: str = Field(min_length=1)

    location: DocumentSourceLocation = Field(
        default_factory=DocumentSourceLocation,
    )


class ParsedDocument(BaseModel):
    """Normalized text extracted from an uploaded document."""

    document_id: str = Field(min_length=1)
    total_pages: int | None = Field(default=None, ge=0)
    pages_processed: int | None = Field(default=None, ge=0)

    blocks: list[ParsedDocumentBlock] = Field(
        default_factory=list,
    )

