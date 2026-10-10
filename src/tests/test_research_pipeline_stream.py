"""Tests for single-run LangGraph event recording and SSE serialization."""

import asyncio

from src.services.research_pipeline_stream_service import (
    PipelineRecorder,
    encode_sse,
    run_graph_with_events,
)


class FakeGraph:
    """Return LangGraph-style v2 events without calling a real LLM."""

    def __init__(self, final_state):
        self.final_state = final_state
        self.calls = 0

    async def astream_events(self, initial_state, *, config, version):
        assert version == "v2"
        assert config == {"recursion_limit": 50}

        self.calls += 1

        yield {
            "event": "on_chain_start",
            "run_id": "root",
            "parent_ids": [],
            "name": "deep_research_job",
            "data": {"input": initial_state},
        }

        yield {
            "event": "on_chain_start",
            "run_id": "planner",
            "parent_ids": ["root"],
            "name": "research_planner",
            "data": {},
        }

        yield {
            "event": "on_chain_end",
            "run_id": "planner",
            "parent_ids": ["root"],
            "name": "research_planner",
            "data": {
                "output": {
                    "private_text": "SECRET",
                }
            },
        }

        yield {
            "event": "on_chain_start",
            "run_id": "router",
            "parent_ids": ["root"],
            "name": "research_routing",
            "data": {},
        }

        yield {
            "event": "on_chain_end",
            "run_id": "router",
            "parent_ids": ["root"],
            "name": "research_routing",
            "data": {
                "output": {
                    "research_control": {
                        "last_decision": {
                            "action": "continue_research",
                            "reason": "A gap remains.",
                        },
                        "pending_gaps": ["gap"],
                    },
                },
            },
        }

        yield {
            "event": "on_chain_end",
            "run_id": "root",
            "parent_ids": [],
            "name": "deep_research_job",
            "data": {"output": self.final_state},
        }


def test_stream_captures_final_state_without_duplicate_execution():
    async def run():
        state = {
            "job_id": "job-1",
            "research": {
                "iteration_count": 0,
            },
        }

        graph = FakeGraph(state)
        events = []

        async def persist(event):
            events.append(event)

        recorder = PipelineRecorder(
            job_id="job-1",
            persist=persist,
        )

        final_state = await run_graph_with_events(
            graph=graph,
            state=state,
            config={"recursion_limit": 50},
            recorder=recorder,
        )

        # The final state must come from the same graph execution.
        assert final_state is state
        assert graph.calls == 1

        assert [event.event_type for event in events] == [
            "workflow_started",
            "stage_started",
            "stage_completed",
            "stage_started",
            "stage_completed",
            "research_retry",
        ]

        # Events must have consecutive sequence numbers.
        assert [event.sequence for event in events] == list(range(1, 7))

        # The research router requested another iteration.
        assert events[-1].iteration == 1
        assert events[-1].details["next_iteration"] == 2
        assert events[-1].details["gap_count"] == 1

        # SSE progress must not expose private graph outputs.
        payload = "".join(encode_sse(event) for event in events)

        assert "SECRET" not in payload
        assert "id: 6\n" in payload
        assert "event: research_retry\n" in payload

    asyncio.run(run())

