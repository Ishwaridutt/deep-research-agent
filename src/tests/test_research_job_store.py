"""Tests for SQLite research jobs, ordered event replay, and restart handling."""

import asyncio

from src.models import ResearchExecutionResult, ResearchExecutionStatus
from src.models.pipeline_event import PipelineEvent
from src.persistence import ResearchJobStore


def test_job_persistence_and_replay(tmp_path):
    async def run():
        store = ResearchJobStore(tmp_path / "research.sqlite3")
        await store.initialize()

        try:
            assert await store.get_job("unknown-job") is None

            await store.create_job(
                job_id="job-1",
                research_session_id="session-1",
                question="Example research question",
                allowed_sources=["web"],
            )

            job = await store.get_job("job-1")
            assert job["status"] == "queued"
            assert job["allowed_sources"] == ["web"]
            assert job["result"] is None

            await store.mark_running("job-1")
            assert (await store.get_job("job-1"))["status"] == "running"

            await store.append_event(PipelineEvent(
                job_id="job-1",
                sequence=1,
                event_type="workflow_started",
                stage="workflow",
                status="running",
            ))

            await store.append_event(PipelineEvent(
                job_id="job-1",
                sequence=2,
                event_type="stage_started",
                stage="research_planner",
                status="running",
                iteration=1,
            ))

            # Reconnection should replay only events after the supplied ID.
            replay = await store.get_events("job-1", after=1)
            assert [event.sequence for event in replay] == [2]

            result = ResearchExecutionResult(
                job_id="job-1",
                research_session_id="session-1",
                status=ResearchExecutionStatus.FAILED,
                stop_reason=None,
                report=None,
                errors=[],
            )

            await store.finish_job(
                result=result,
                terminal_event=PipelineEvent(
                    job_id="job-1",
                    sequence=3,
                    event_type="workflow_failed",
                    stage="workflow",
                    status="failed",
                ),
                error_message="Test execution failed.",
            )

            job = await store.get_job("job-1")

            assert job["status"] == "failed"
            assert job["result"]["job_id"] == "job-1"
            assert job["error_message"] == "Test execution failed."

            events = await store.get_events("job-1")

            assert [event.sequence for event in events] == [1, 2, 3]
            assert events[-1].event_type == "workflow_failed"

            assert [
                event.sequence
                for event in await store.get_events("job-1", after=3)
            ] == []

        finally:
            await store.close()

    asyncio.run(run())


def test_restart_marks_unfinished_job_failed(tmp_path):
    async def run():
        path = tmp_path / "research.sqlite3"

        store = ResearchJobStore(path)
        await store.initialize()

        try:
            await store.create_job(
                job_id="unfinished",
                research_session_id="session-1",
                question="Example research question",
                allowed_sources=["web"],
            )
        finally:
            await store.close()

        # Restarting cannot resume work until LangGraph checkpointing exists.
        recovered = ResearchJobStore(path)
        await recovered.initialize()

        try:
            job = await recovered.get_job("unfinished")

            assert job["status"] == "failed"
            assert "restart" in job["error_message"].lower()
            assert job["completed_at"] is not None

        finally:
            await recovered.close()

    asyncio.run(run())

    