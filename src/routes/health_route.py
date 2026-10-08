"""Application health-check route."""

from fastapi import (
    APIRouter,
)


router = APIRouter(
    tags=["health"],
)


@router.get("/health")
async def health() -> dict[str, str]:
    """
    Returns basic process health.

    This intentionally does not call external providers such as the LLM
    or Tavily. It answers whether the application process is alive.
    """

    return {
        "status": "running",
    }

