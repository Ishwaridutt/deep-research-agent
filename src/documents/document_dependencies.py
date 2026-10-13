"""Document dependency construction with explicit resource scopes."""

from fastapi import Request

from src.config import Settings
from src.documents.document_parser import DocumentParser
from src.documents.document_registry import DocumentRegistry
from src.documents.local_document_file_store import LocalDocumentFileStore
from src.services.document_service import DocumentService


def create_document_service(
    *,
    settings: Settings,
) -> DocumentService:
    """Build reusable document upload and parsing dependencies."""

    return DocumentService(
        registry=DocumentRegistry(),
        file_store=LocalDocumentFileStore(
            root_directory=settings.documents.upload_directory,
        ),
        parser=DocumentParser(),
        settings=settings.documents,
    )


def get_document_service(request: Request) -> DocumentService:
    """Return the application-scoped private-document service."""

    return request.app.state.document_service

