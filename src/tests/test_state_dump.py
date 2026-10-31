"""Offline tests for the debug state dumps written after the coverage and finalize nodes."""

import asyncio
import json

from src.graph.graph_state import create_initial_workflow_state
from src.graph.state_dump import RESEARCH_STATE_FILE, run_node_and_dump_state
from src.models import ResearchSource, ResearchStopReason, SubQuestion


def initial_state():
    return create_initial_workflow_state(
        job_id="job-1", research_session_id="session-1", question="Explain the findings",
        allowed_sources=[ResearchSource.WEB],
    )


async def coverage_round(state):
    return {"research": {**state["research"], "iteration_count": state["research"]["iteration_count"] + 1}}


def finalize(state):
    return {"writer": {**state["writer"], "final_markdown": "# Report"}}


def test_dump_contains_the_state_after_the_node_and_each_round_overwrites_it(tmp_path):
    state = initial_state()
    state["research"]["subquestions"] = [SubQuestion(id="SQ-1", question="What?", objective="Find out.")]
    state["research"]["stop_reason"] = ResearchStopReason.COVERAGE_SUFFICIENT

    for _ in range(2):
        update = asyncio.run(run_node_and_dump_state(
            state, node=coverage_round, directory=tmp_path, file_name=RESEARCH_STATE_FILE,
        ))
        state = {**state, **update}

    dumped = json.loads((tmp_path / "job-1" / RESEARCH_STATE_FILE).read_text())
    assert dumped["research"]["iteration_count"] == 2
    assert dumped["research"]["subquestions"][0]["id"] == "SQ-1"
    assert dumped["research"]["stop_reason"] == "coverage_sufficient"
    assert dumped["question"] == "Explain the findings"


def test_sync_nodes_are_supported_and_the_update_is_returned_unchanged(tmp_path):
    state = initial_state()
    update = asyncio.run(run_node_and_dump_state(state, node=finalize, directory=tmp_path, file_name="final_state.json"))
    assert update == finalize(state)
    dumped = json.loads((tmp_path / "job-1" / "final_state.json").read_text())
    assert dumped["writer"]["final_markdown"] == "# Report" and "research" in dumped


def test_a_failed_dump_never_fails_the_node(tmp_path, caplog):
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("file")
    with caplog.at_level("WARNING"):
        update = asyncio.run(run_node_and_dump_state(
            initial_state(), node=finalize, directory=blocker, file_name="final_state.json",
        ))
    assert update["writer"]["final_markdown"] == "# Report"
    assert any(getattr(record, "event", "") == "state_dump_failed" for record in caplog.records)
