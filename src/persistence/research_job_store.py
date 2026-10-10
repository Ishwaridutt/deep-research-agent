"""SQLite storage for asynchronous research jobs and their ordered events."""

import asyncio
import json
from pathlib import Path

import aiosqlite

from src.models import ResearchExecutionResult
from src.models.pipeline_event import PipelineEvent


class ResearchJobStore:
    """One SQLite connection per API process; supports replay and job lookup."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    def _connection(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Research job store is not initialized.")
        return self._db

    async def initialize(self) -> None:
        """Create tables and mark unfinished jobs from prior process runs failed."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.path)
        self._db.row_factory = aiosqlite.Row

        try:
            await self._db.execute("PRAGMA journal_mode=WAL")
            await self._db.execute("PRAGMA foreign_keys=ON")
            await self._db.execute("PRAGMA busy_timeout=5000")
            await self._db.executescript("""
                CREATE TABLE IF NOT EXISTS research_jobs (
                    job_id TEXT PRIMARY KEY,
                    research_session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    allowed_sources TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (
                        status IN ('queued', 'running', 'completed', 'failed')
                    ),
                    result_json TEXT,
                    error_message TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    started_at TEXT,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS research_events (
                    job_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    PRIMARY KEY (job_id, sequence),
                    FOREIGN KEY (job_id) REFERENCES research_jobs(job_id)
                );
            """)

            # There is no LangGraph checkpointer yet, so jobs cannot resume.
            await self._db.execute("""
                UPDATE research_jobs
                SET status = 'failed',
                    error_message = 'Research interrupted by server restart.',
                    completed_at = CURRENT_TIMESTAMP
                WHERE status IN ('queued', 'running')
            """)
            await self._db.commit()
        except Exception:
            await self.close()
            raise

    async def create_job(
        self,
        *,
        job_id: str,
        research_session_id: str,
        question: str,
        allowed_sources: list[str],
    ) -> None:
        async with self._lock:
            db = self._connection()
            await db.execute("""
                INSERT INTO research_jobs (
                    job_id, research_session_id, question, allowed_sources, status
                ) VALUES (?, ?, ?, ?, 'queued')
            """, (job_id, research_session_id, question, json.dumps(allowed_sources)))
            await db.commit()

    async def mark_running(self, job_id: str) -> None:
        async with self._lock:
            db = self._connection()
            await db.execute("""
                UPDATE research_jobs
                SET status = 'running', started_at = CURRENT_TIMESTAMP
                WHERE job_id = ? AND status = 'queued'
            """, (job_id,))
            await db.commit()

    async def append_event(self, event: PipelineEvent) -> None:
        async with self._lock:
            db = self._connection()
            await db.execute("""
                INSERT INTO research_events (job_id, sequence, event_json)
                VALUES (?, ?, ?)
            """, (event.job_id, event.sequence, event.model_dump_json()))
            await db.commit()

    async def finish_job(
        self,
        *,
        result: ResearchExecutionResult,
        terminal_event: PipelineEvent,
        error_message: str | None = None,
    ) -> None:
        """Write terminal status and terminal SSE event in one transaction."""

        async with self._lock:
            db = self._connection()
            try:
                await db.execute("""
                    UPDATE research_jobs
                    SET status = ?, result_json = ?, error_message = ?,
                        completed_at = CURRENT_TIMESTAMP
                    WHERE job_id = ?
                """, (
                    result.status.value,
                    result.model_dump_json(),
                    error_message,
                    result.job_id,
                ))
                await db.execute("""
                    INSERT INTO research_events (job_id, sequence, event_json)
                    VALUES (?, ?, ?)
                """, (
                    terminal_event.job_id,
                    terminal_event.sequence,
                    terminal_event.model_dump_json(),
                ))
                await db.commit()
            except Exception:
                await db.rollback()
                raise

    async def get_job(self, job_id: str) -> dict | None:
        async with self._lock:
            db = self._connection()
            async with db.execute(
                "SELECT * FROM research_jobs WHERE job_id = ?", (job_id,)
            ) as cursor:
                row = await cursor.fetchone()

        if row is None:
            return None

        job = dict(row)
        job["allowed_sources"] = json.loads(job["allowed_sources"])
        result_json = job.pop("result_json")
        job["result"] = json.loads(result_json) if result_json else None
        return job

    async def get_events(
        self,
        job_id: str,
        *,
        after: int = 0,
        limit: int = 100,
    ) -> list[PipelineEvent]:
        async with self._lock:
            db = self._connection()
            async with db.execute("""
                SELECT event_json FROM research_events
                WHERE job_id = ? AND sequence > ?
                ORDER BY sequence ASC LIMIT ?
            """, (job_id, after, limit)) as cursor:
                rows = await cursor.fetchall()

        return [
            PipelineEvent.model_validate_json(row["event_json"])
            for row in rows
        ]

    async def close(self) -> None:
        async with self._lock:
            if self._db is not None:
                await self._db.close()
                self._db = None


