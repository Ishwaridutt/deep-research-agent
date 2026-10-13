"""Routes for private document management."""

from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Header,
    HTTPException,
    UploadFile,
    status,
)

from src.documents.document_dependencies import get_document_service
from src.documents.document_exceptions import (
    DocumentFileTooLargeError,
    DocumentQuotaExceededError,
    EmptyDocumentError,
    UnsupportedDocumentFormatError,
)
from src.models.document_models import DocumentRecord
from src.services.document_service import DocumentService


router = APIRouter(
    prefix="/documents",
    tags=["documents"],
)


def get_owner_id(
    x_owner_id: Annotated[
        str,
        Header(alias="X-Owner-Id"),
    ],
) -> str:
    """
    Temporary owner identification.

    Replace with authenticated user identity later.
    """

    return x_owner_id


@router.get("", response_model=list[DocumentRecord])
async def list_documents(
    owner_id: Annotated[str, Depends(get_owner_id)],
    document_service: Annotated[
        DocumentService,
        Depends(get_document_service),
    ],
) -> list[DocumentRecord]:
    """List uploaded documents available to the requesting owner."""

    return await document_service.list_documents(owner_id)


@router.post(
    "",
    response_model=DocumentRecord,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    owner_id: Annotated[
        str,
        Depends(get_owner_id),
    ],
    document_service: Annotated[
        DocumentService,
        Depends(get_document_service),
    ],
    file: UploadFile = File(...),
) -> DocumentRecord:
    """Upload a private document."""

    try:
        return await document_service.upload_document(
            owner_id=owner_id,
            filename=file.filename or "",
            content_type=file.content_type or "application/octet-stream",
            source=file.file,
        )

    except UnsupportedDocumentFormatError as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc

    except EmptyDocumentError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except (DocumentQuotaExceededError, DocumentFileTooLargeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=str(exc),
        ) from exc

    finally:
        await file.close()
