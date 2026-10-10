"""Route serving the frontend application entry page."""

from pathlib import Path

from fastapi import (
    APIRouter,
)
from fastapi.responses import (
    FileResponse,
)


_FRONTEND_DIRECTORY = Path(__file__).resolve().parent.parent/ "static"


_INDEX_FILE = (
    _FRONTEND_DIRECTORY
    / "index.html"
)


router = APIRouter(
    tags=["frontend"],
)


@router.get(
    "/",
    response_class=FileResponse,
    include_in_schema=False,
)
async def index() -> FileResponse:
    """
    Serves the browser application's entry HTML file.
    """

    return FileResponse(
        path=_INDEX_FILE,
        media_type="text/html",
    )

