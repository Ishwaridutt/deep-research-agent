"""Focused offline tests for private documents using the shared retrieval pipeline."""

import asyncio
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config.settings import DocumentSettings
from src.documents.document_registry import DocumentRegistry
from src.models.document_models import (
    DocumentFormat,
    DocumentRecord,
    DocumentSourceLocation,
    DocumentStatus,
    ParsedDocument,
    ParsedDocumentBlock,
    StoredDocumentFile,
)
from src.models.retrieval_model import RetrievalChunk, RetrievalDocument
from src.retrieval.chunking.retrieval_chunker import RetrievalChunker
from src.services.document_service import DocumentService


class FakeGeneratedChunker:
    """Return one deterministic generated chunk for each retrieval document."""

    def __call__(self, text: str) -> list[SimpleNamespace]:
        """Create the subset of a Chonkie chunk used by RetrievalChunker."""

        return [
            SimpleNamespace(
                text=text[1:],
                start_index=1,
                end_index=len(text),
                token_count=max(1, len(text.split())),
            )
        ]


def create_test_chunker() -> RetrievalChunker:
    """Create a RetrievalChunker without loading a real tokenizer."""

    chunker = object.__new__(RetrievalChunker)
    chunker.chunker = FakeGeneratedChunker()
    return chunker


def create_document_record(
    *,
    document_id: str = "document-1",
    owner_id: str = "owner-1",
) -> DocumentRecord:
    """Create stable uploaded-document metadata for pipeline tests."""

    return DocumentRecord(
        document_id=document_id,
        owner_id=owner_id,
        filename=f"{document_id}.txt",
        document_format=DocumentFormat.TXT,
        content_type="text/plain",
        size_bytes=11,
        content_hash="0" * 64,
        storage_key=f"{document_id}.txt",
    )


class FakeFileStore:
    """Capture file operations without touching the filesystem."""

    def __init__(self) -> None:
        """Initialize captured save and deletion operations."""

        self.saved_storage_keys: list[str] = []
        self.deleted_storage_keys: list[str] = []

    async def save(self, *, source, storage_key: str, max_bytes: int):
        """Return deterministic stored-file metadata."""

        self.saved_storage_keys.append(storage_key)
        return StoredDocumentFile(
            storage_key=storage_key,
            size_bytes=4,
            content_hash="0" * 64,
        )

    def get_path(self, *, storage_key: str) -> Path:
        """Return a harmless path for parser fakes."""

        return Path(storage_key)

    async def delete(self, *, storage_key: str) -> None:
        """Capture source-file cleanup."""

        self.deleted_storage_keys.append(storage_key)


class FakeParser:
    """Return configured parsed documents without reading real files."""

    def __init__(self, parsed_by_id=None, error: Exception | None = None):
        """Store deterministic parse results and an optional failure."""

        self.parsed_by_id = parsed_by_id or {}
        self.error = error
        self.parse_calls: list[str] = []

    def parse(self, document_id, path, document_format):
        """Return the parsed document configured for the supplied ID."""

        self.parse_calls.append(document_id)
        if self.error is not None:
            raise self.error
        return self.parsed_by_id[document_id]


class FakeRetrievalSession:
    """Capture documents delegated to the existing retrieval session boundary."""

    def __init__(self, error: Exception | None = None) -> None:
        """Initialize an empty job-scoped session."""

        self.error = error
        self.build_calls = 0
        self.documents: list[RetrievalDocument] = []
        self.chunks: list[RetrievalChunk] = []
        self.cleared = False

    def build(self, documents: list[RetrievalDocument]) -> None:
        """Represent existing retrieval chunking and indexing with offline data."""

        self.build_calls += 1
        self.documents = list(documents)
        if self.error is not None:
            raise self.error

        self.chunks = [
            RetrievalChunk(
                chunk_id=f"{document.document_id}:0",
                document_id=document.document_id,
                content=document.content,
                source_type=document.source_type,
                chunk_index=0,
                token_count=1,
                title=document.title,
                section=document.metadata.get("section_title"),
                metadata=document.metadata.copy(),
            )
            for document in documents
        ]

    def clear(self) -> None:
        """Capture retrieval-session cleanup after a failed build."""

        self.cleared = True
        self.chunks = []


def create_service(
    *,
    registry: DocumentRegistry,
    parser: FakeParser,
    file_store: FakeFileStore | None = None,
) -> DocumentService:
    """Create a document service without a separate chunker or vector index."""

    return DocumentService(
        registry=registry,
        file_store=file_store or FakeFileStore(),
        parser=parser,
        settings=DocumentSettings(),
    )


def test_upload_only_saves_file_and_metadata():
    """Defer parsing and retrieval work until a research job selects the file."""

    async def run() -> None:
        """Upload through offline storage and parser fakes."""

        registry = DocumentRegistry()
        file_store = FakeFileStore()
        parser = FakeParser()
        service = create_service(
            registry=registry,
            parser=parser,
            file_store=file_store,
        )

        document = await service.upload_document(
            owner_id="owner-1",
            filename="notes.txt",
            content_type="text/plain",
            source=BytesIO(b"test"),
        )

        assert file_store.saved_storage_keys == [document.storage_key]
        assert parser.parse_calls == []
        assert document.status == DocumentStatus.UPLOADED
        assert document.chunk_count == 0

    asyncio.run(run())


def test_document_service_lists_only_owner_documents_newest_first():
    async def run() -> None:
        registry = DocumentRegistry()
        older = create_document_record(document_id="older")
        newer = create_document_record(document_id="newer").model_copy(
            update={"created_at": datetime(2026, 1, 2, tzinfo=UTC)}
        )
        older = older.model_copy(
            update={"created_at": datetime(2026, 1, 1, tzinfo=UTC)}
        )
        other_owner = create_document_record(
            document_id="private",
            owner_id="owner-2",
        )
        for document in (older, newer, other_owner):
            await registry.save_document(document)

        service = create_service(registry=registry, parser=FakeParser())

        assert [item.document_id for item in await service.list_documents("owner-1")] == [
            "newer",
            "older",
        ]

    asyncio.run(run())


def test_document_service_builds_existing_retrieval_session_once():
    """Delegate selected parsed blocks to one job-scoped retrieval session."""

    async def run() -> None:
        """Prepare two document blocks with complete provenance."""

        registry = DocumentRegistry()
        document = create_document_record()
        await registry.save_document(document)
        parser = FakeParser({
            document.document_id: ParsedDocument(
                document_id=document.document_id,
                blocks=[
                    ParsedDocumentBlock(
                        text="alpha",
                        location=DocumentSourceLocation(
                            page_start=1,
                            page_end=1,
                            section_title="Start",
                        ),
                    ),
                    ParsedDocumentBlock(
                        text="beta",
                        location=DocumentSourceLocation(
                            page_start=2,
                            page_end=2,
                            section_title="End",
                        ),
                    ),
                ],
            )
        })
        session = FakeRetrievalSession()
        service = create_service(registry=registry, parser=parser)

        await service.build_document_retrieval_session(
            owner_id=document.owner_id,
            document_ids=[document.document_id, document.document_id],
            session=session,
        )

        assert session.build_calls == 1
        assert parser.parse_calls == [document.document_id]
        assert [item.document_id for item in session.documents] == [
            "document-1:block:0",
            "document-1:block:1",
        ]
        assert session.documents[0].metadata == {
            "owner_id": "owner-1",
            "private_document_id": "document-1",
            "filename": "document-1.txt",
            "block_index": 0,
            "page_start": 1,
            "page_end": 1,
            "section_title": "Start",
            "char_start": 0,
            "char_end": 5,
        }
        assert session.documents[1].metadata["char_start"] == 7
        assert session.documents[1].metadata["char_end"] == 11

        saved = await registry.get_document(
            owner_id=document.owner_id,
            document_id=document.document_id,
        )
        assert saved.status == DocumentStatus.READY
        assert saved.chunk_count == 2

    asyncio.run(run())


def test_document_service_rejects_another_owners_document():
    """Enforce ownership before parsing or building the retrieval session."""

    async def run() -> None:
        """Attempt to select a document belonging to another owner."""

        registry = DocumentRegistry()
        document = create_document_record(owner_id="owner-2")
        await registry.save_document(document)
        parser = FakeParser()
        session = FakeRetrievalSession()
        service = create_service(registry=registry, parser=parser)

        with pytest.raises(ValueError, match="Document not found"):
            await service.build_document_retrieval_session(
                owner_id="owner-1",
                document_ids=[document.document_id],
                session=session,
            )

        assert parser.parse_calls == []
        assert session.build_calls == 0

    asyncio.run(run())


def test_document_service_clears_session_and_records_failure():
    """Clear partial retrieval data and mark documents failed on build errors."""

    async def run() -> None:
        """Fail at the shared retrieval-session build boundary."""

        registry = DocumentRegistry()
        document = create_document_record()
        await registry.save_document(document)
        parser = FakeParser({
            document.document_id: ParsedDocument(
                document_id=document.document_id,
                blocks=[ParsedDocumentBlock(text="alpha")],
            )
        })
        session = FakeRetrievalSession(error=RuntimeError("index failed"))
        service = create_service(registry=registry, parser=parser)

        with pytest.raises(RuntimeError, match="index failed"):
            await service.build_document_retrieval_session(
                owner_id=document.owner_id,
                document_ids=[document.document_id],
                session=session,
            )

        assert session.cleared is True
        saved = await registry.get_document(
            owner_id=document.owner_id,
            document_id=document.document_id,
        )
        assert saved.status == DocumentStatus.FAILED
        assert saved.error_message == "index failed"

    asyncio.run(run())


def test_retrieval_chunker_preserves_document_provenance():
    """Apply block offsets and section metadata inside the shared chunker."""

    chunks = create_test_chunker().chunk_document(
        RetrievalDocument(
            document_id="document-1:block:0",
            content="alpha",
            source_type="document",
            metadata={
                "private_document_id": "document-1",
                "section_title": "Start",
                "char_start": 10,
            },
        )
    )

    assert chunks[0].content == "lpha"
    assert chunks[0].section == "Start"
    assert chunks[0].metadata["private_document_id"] == "document-1"
    assert chunks[0].metadata["char_start"] == 11
    assert chunks[0].metadata["char_end"] == 15


def test_document_service_deletes_upload_but_retains_metadata():
    """Delete only the uploaded source so citation metadata remains available."""

    async def run() -> None:
        """Exercise optional source-file cleanup with an offline fake."""

        registry = DocumentRegistry()
        document = create_document_record()
        await registry.save_document(document)
        file_store = FakeFileStore()
        service = create_service(
            registry=registry,
            parser=FakeParser(),
            file_store=file_store,
        )

        await service.delete_uploaded_document(
            owner_id=document.owner_id,
            document_id=document.document_id,
        )

        assert file_store.deleted_storage_keys == [document.storage_key]
        assert await registry.get_document(
            owner_id=document.owner_id,
            document_id=document.document_id,
        ) == document

    asyncio.run(run())
