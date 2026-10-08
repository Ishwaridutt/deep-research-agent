"""HTTP route for executing research jobs."""

from fastapi import (
    APIRouter,
    status,
)

from src.models import (
    ResearchExecutionResult,
    ResearchRequest,
)
from src.services import ResearchService


def create_research_router(
    *,
    research_service: ResearchService,
) -> APIRouter:
    """
    Creates the research API router with its application service injected.

    The route layer knows only about HTTP concerns and ResearchService.
    It does not construct graph state or invoke LangGraph directly.
    """

    router = APIRouter(
        prefix="/api",
        tags=["research"],
    )

    @router.post(
        "/research",
        response_model=ResearchExecutionResult,
        status_code=status.HTTP_200_OK,
    )
    async def research(
        request: ResearchRequest,
    ) -> ResearchExecutionResult:
        """
        Executes one complete deep-research job.
        """

        return await research_service.research(
            question=request.question,
            allowed_sources=request.sources,
        )

    return router

