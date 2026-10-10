"""HTTP routes for synchronous and persisted research jobs."""

import asyncio

from fastapi import APIRouter, Header, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse

from src.models import ResearchExecutionResult, ResearchRequest
from src.models.pipeline_event import ResearchJobCreated, ResearchJobStatusResponse
from src.services.research_pipeline_stream_service import encode_sse
from src.services.research_service import ResearchJobQueueFull, ResearchService


TERMINAL_STATUSES = {"completed", "failed"}


def create_research_router(*, research_service: ResearchService) -> APIRouter:
    """Create routes while keeping all workflow execution in ResearchService."""

    router = APIRouter(prefix="/api", tags=["research"])

    @router.post(
        "/research",
        response_model=ResearchExecutionResult,
        status_code=status.HTTP_200_OK,
    )
    async def research(request: ResearchRequest) -> ResearchExecutionResult:
        """Existing endpoint: run research and return its final result."""

        return await research_service.research(
            question=request.question,
            allowed_sources=request.sources,
        )

    @router.post(
        "/research/jobs",
        response_model=ResearchJobCreated,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def create_research_job(request: ResearchRequest) -> ResearchJobCreated:
        """Queue a persistent research job; return its ID immediately."""

        try:
            return await research_service.start_job(
                question=request.question,
                allowed_sources=request.sources,
            )
        except ResearchJobQueueFull as exc:
            raise HTTPException(status_code=429, detail=str(exc)) from exc


    @router.get(
        "/research/jobs/status/{job_id}",
        response_model=ResearchJobStatusResponse,
    )
    async def get_research_job_status(job_id: str) -> ResearchJobStatusResponse:
        """Return the persisted lifecycle state and optional final result."""

        job = await research_service.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Research job not found.")
        return ResearchJobStatusResponse.model_validate(job)


    @router.get("/research/jobs/{job_id}/events")
    async def stream_research_job_events(
        job_id: str,
        request: Request,
        after: int = Query(default=0, ge=0),
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        """Replay stored events and stream new events as SSE."""

        if await research_service.get_job(job_id) is None:
            raise HTTPException(status_code=404, detail="Research job not found.")

        if last_event_id is not None:
            try:
                cursor = int(last_event_id)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail="Last-Event-ID must be a nonnegative integer.",
                ) from exc
            if cursor < 0:
                raise HTTPException(
                    status_code=400,
                    detail="Last-Event-ID must be a nonnegative integer.",
                )
            after = max(after, cursor)

        async def generate():
            sequence = after
            loop = asyncio.get_running_loop()
            last_keepalive = loop.time()

            while not await request.is_disconnected():
                events = await research_service.get_events(job_id, after=sequence)
                for event in events:
                    sequence = event.sequence
                    yield encode_sse(event)

                # Terminal event and result are written atomically. Drain
                # all batches before closing the stream.
                job = await research_service.get_job(job_id)
                if job is None or (job["status"] in TERMINAL_STATUSES and not events):
                    break

                if loop.time() - last_keepalive >= 15:
                    yield ": keep-alive\n\n"
                    last_keepalive = loop.time()

                await asyncio.sleep(0.5)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return router

