import asyncio
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO
from uuid import uuid4

from src.config.settings import DocumentSettings
from src.documents.document_exceptions import UnsupportedDocumentFormatError
from src.documents.document_registry import DocumentRegistry
from src.documents.local_document_file_store import LocalDocumentFileStore
from src.models.document_models import (
    DocumentFormat,
    DocumentRecord,
    DocumentStatus,
    ParsedDocument,
    utc_now,
)
from src.models.retrieval_model import RetrievalDocument
from src.retrieval.retrieval_session import RetrievalSession

if TYPE_CHECKING:
    from src.documents.document_parser import DocumentParser


FORMAT_MAP = {
    ".pdf": DocumentFormat.PDF,
    ".docx": DocumentFormat.DOCX,
    ".txt": DocumentFormat.TXT,
    ".md": DocumentFormat.MARKDOWN,
}


class DocumentService:
    """Store uploads and prepare selected files for the retrieval pipeline."""

    def __init__(
        self,
        registry: DocumentRegistry,
        file_store: LocalDocumentFileStore,
        parser: "DocumentParser",
        settings: DocumentSettings,
    ):
        """Store reusable upload and parsing dependencies."""

        self.registry = registry
        self.file_store = file_store
        self.parser = parser
        self.settings = settings

    async def upload_document(
        self,
        owner_id: str,
        filename: str,
        content_type: str,
        source: BinaryIO,
    ) -> DocumentRecord:
        """Validate and store one uploaded private document."""

        extension = Path(filename).suffix.lower()

        if extension not in FORMAT_MAP:
            raise UnsupportedDocumentFormatError(
                "Unsupported document format."
            )

        current_size = await self.registry.get_total_size(
            owner_id=owner_id
        )

        remaining = self.settings.max_total_size_bytes - current_size

        document_id = str(uuid4())
        storage_key = f"{document_id}{extension}"

        stored = await self.file_store.save(
            source=source,
            storage_key=storage_key,
            max_bytes=remaining,
        )

        document = DocumentRecord(
            document_id=document_id,
            owner_id=owner_id,
            filename=filename,
            document_format=FORMAT_MAP[extension],
            content_type=content_type,
            size_bytes=stored.size_bytes,
            content_hash=stored.content_hash,
            storage_key=storage_key,
        )

        await self.registry.save_document(document)

        return document

    async def list_documents(self, owner_id: str) -> list[DocumentRecord]:
        """List the requesting owner's uploads, newest first."""

        documents = await self.registry.list_documents(owner_id=owner_id)
        return sorted(documents, key=lambda item: item.created_at, reverse=True)

    async def get_document(
        self,
        owner_id: str,
        document_id: str,
    ) -> DocumentRecord:
        """Return one document owned by the requesting user."""

        document = await self.registry.get_document(
            owner_id=owner_id,
            document_id=document_id,
        )

        if not document:
            raise ValueError("Document not found.")

        return document

    async def parse_document(
        self,
        owner_id: str,
        document_id: str,
    ) -> ParsedDocument:
        """Parse a stored document without blocking the event loop."""

        document = await self.get_document(
            owner_id,
            document_id,
        )

        return await self._parse_document_record(document)

    async def build_document_retrieval_session(
        self,
        owner_id: str,
        document_ids: list[str],
        session: RetrievalSession,
    ) -> None:
        """Parse selected files and build the existing job retrieval session."""

        selected_document_ids = list(dict.fromkeys(document_ids))
        documents = [
            await self.get_document(owner_id, document_id)
            for document_id in selected_document_ids
        ]

        for document in documents:
            await self._save_processing_state(
                document,
                status=DocumentStatus.PROCESSING,
                chunk_count=0,
                error_message=None,
            )

        try:
            retrieval_documents: list[RetrievalDocument] = []
            for document in documents:
                parsed_document = await self._parse_document_record(document)
                retrieval_documents.extend(
                    self._create_retrieval_documents(
                        document,
                        parsed_document,
                    )
                )

            # RetrievalSession owns chunking, semantic indexing, BM25 indexing,
            # and hybrid ranking for both web and private-document sources.
            await asyncio.to_thread(
                session.build,
                retrieval_documents,
            )
        except Exception as exc:
            session.clear()
            for document in documents:
                await self._save_processing_state(
                    document,
                    status=DocumentStatus.FAILED,
                    chunk_count=0,
                    error_message=str(exc),
                )
            raise

        chunks_per_document = Counter(
            chunk.metadata.get("private_document_id")
            for chunk in session.chunks
        )
        for document in documents:
            await self._save_processing_state(
                document,
                status=DocumentStatus.READY,
                chunk_count=chunks_per_document[document.document_id],
                error_message=None,
            )

        # Optional cleanup: the original upload is no longer needed after
        # successful parsing and indexing. Uncomment to enable file deletion.
        # for document_id in selected_document_ids:
        #     await self.delete_uploaded_document(owner_id, document_id)

    async def delete_uploaded_document(
        self,
        owner_id: str,
        document_id: str,
    ) -> None:
        """Delete an uploaded source file while retaining its metadata record."""

        document = await self.get_document(owner_id, document_id)
        await self.file_store.delete(storage_key=document.storage_key)

    def _create_retrieval_documents(
        self,
        document: DocumentRecord,
        parsed_document: ParsedDocument,
    ) -> list[RetrievalDocument]:
        """Convert parsed blocks into inputs for the existing retrieval session."""

        retrieval_documents: list[RetrievalDocument] = []
        next_char_start = 0

        for block_index, block in enumerate(parsed_document.blocks):
            block_char_start = (
                block.location.char_start
                if block.location.char_start is not None
                else next_char_start
            )
            block_char_end = (
                block.location.char_end
                if block.location.char_end is not None
                else block_char_start + len(block.text)
            )
            retrieval_documents.append(
                RetrievalDocument(
                    # Block IDs keep chunk IDs unique while the uploaded
                    # document ID remains available for citations.
                    document_id=(
                        f"{document.document_id}:block:{block_index}"
                    ),
                    content=block.text,
                    source_type="document",
                    title=document.filename,
                    metadata={
                        "owner_id": document.owner_id,
                        "private_document_id": document.document_id,
                        "filename": document.filename,
                        "block_index": block_index,
                        "page_start": block.location.page_start,
                        "page_end": block.location.page_end,
                        "section_title": block.location.section_title,
                        "char_start": block_char_start,
                        "char_end": block_char_end,
                    },
                )
            )

            next_char_start = max(next_char_start, block_char_end + 2)

        return retrieval_documents

    async def _parse_document_record(
        self,
        document: DocumentRecord,
    ) -> ParsedDocument:
        """Parse an already-authorized document record off the event loop."""

        path = self.file_store.get_path(
            storage_key=document.storage_key
        )
        if not path:
            raise ValueError("Stored document not found.")

        return await asyncio.to_thread(
            self.parser.parse,
            document.document_id,
            path,
            document.document_format,
        )

    async def _save_processing_state(
        self,
        document: DocumentRecord,
        *,
        status: DocumentStatus,
        chunk_count: int,
        error_message: str | None,
    ) -> None:
        """Persist the latest document-processing state in the registry."""

        updated_document = document.model_copy(
            update={
                "status": status,
                "chunk_count": chunk_count,
                "error_message": error_message,
                "updated_at": utc_now(),
            }
        )
        await self.registry.save_document(updated_document)
