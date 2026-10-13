"""In-memory registry for uploaded document metadata and status."""

import asyncio

from src.models.document_models import (
    DocumentProcessingEvent,
    DocumentRecord,
)


class DocumentRegistry:
    """Tracks uploaded documents and their latest processing state."""

    def __init__(self) -> None:
        self._documents: dict[
            tuple[str, str],
            DocumentRecord,
        ] = {}

        self._progress: dict[
            tuple[str, str],
            DocumentProcessingEvent,
        ] = {}

        self._lock = asyncio.Lock()

    async def save_document(
        self,
        document: DocumentRecord,
    ) -> None:
        async with self._lock:
            self._documents[
                (document.owner_id, document.document_id)
            ] = document

    async def get_document(
        self,
        *,
        owner_id: str,
        document_id: str,
    ) -> DocumentRecord | None:
        async with self._lock:
            return self._documents.get(
                (owner_id, document_id)
            )

    async def list_documents(
        self,
        *,
        owner_id: str,
    ) -> list[DocumentRecord]:
        async with self._lock:
            return [
                document
                for (stored_owner_id, _), document
                in self._documents.items()
                if stored_owner_id == owner_id
            ]

    async def get_total_size(
        self,
        *,
        owner_id: str,
    ) -> int:
        async with self._lock:
            return sum(
                document.size_bytes
                for (stored_owner_id, _), document
                in self._documents.items()
                if stored_owner_id == owner_id
            )

    async def save_progress(
        self,
        event: DocumentProcessingEvent,
    ) -> None:
        async with self._lock:
            self._progress[
                (event.owner_id, event.document_id)
            ] = event

    async def get_progress(
        self,
        *,
        owner_id: str,
        document_id: str,
    ) -> DocumentProcessingEvent | None:
        async with self._lock:
            return self._progress.get(
                (owner_id, document_id)
            )

    async def delete_document(
        self,
        *,
        owner_id: str,
        document_id: str,
    ) -> None:
        key = (owner_id, document_id)

        async with self._lock:
            self._documents.pop(key, None)
            self._progress.pop(key, None)

            