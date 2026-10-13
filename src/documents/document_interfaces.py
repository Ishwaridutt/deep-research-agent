"""Interfaces for private document storage and retrieval."""

from pathlib import Path
from typing import BinaryIO, Protocol

from src.models.document_models import (
    DocumentCollection,
    DocumentProcessingEvent,
    DocumentRecord,
    StoredDocumentFile,
)


class DocumentMetadataStore(Protocol):
    """Storage contract for document metadata."""

    async def save(
        self,
        document: DocumentRecord,
    ) -> None:
        ...

    async def get(
        self,
        *,
        owner_id: str,
        document_id: str,
    ) -> DocumentRecord | None:
        ...

    async def list_documents(
        self,
        *,
        owner_id: str,
    ) -> list[DocumentRecord]:
        ...

    async def get_total_size(
        self,
        *,
        owner_id: str,
    ) -> int:
        ...

    async def delete(
        self,
        *,
        owner_id: str,
        document_id: str,
    ) -> None:
        ...


class DocumentFileStore(Protocol):
    """Storage contract for original uploaded document files."""

    async def save(
        self,
        *,
        source: BinaryIO,
        storage_key: str,
        max_bytes: int,
    ) -> StoredDocumentFile:
        ...

    def get_path(
        self,
        *,
        storage_key: str,
    ) -> Path | None:
        ...

    async def delete(
        self,
        *,
        storage_key: str,
    ) -> None:
        ...


class DocumentCollectionStore(Protocol):
    """Storage contract for reusable document collections."""

    async def save(
        self,
        collection: DocumentCollection,
    ) -> None:
        ...

    async def get(
        self,
        *,
        owner_id: str,
        collection_id: str,
    ) -> DocumentCollection | None:
        ...

    async def list_collections(
        self,
        *,
        owner_id: str,
    ) -> list[DocumentCollection]:
        ...

    async def delete(
        self,
        *,
        owner_id: str,
        collection_id: str,
    ) -> None:
        ...


class DocumentProgressStore(Protocol):
    """Storage contract for document-processing progress."""

    async def publish(
        self,
        event: DocumentProcessingEvent,
    ) -> None:
        ...

    async def get_latest(
        self,
        *,
        owner_id: str,
        document_id: str,
    ) -> DocumentProcessingEvent | None:
        ...

    async def delete(
        self,
        *,
        owner_id: str,
        document_id: str,
    ) -> None:
        ...

        
