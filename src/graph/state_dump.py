"""Debug dumps of the LangGraph state to JSON files, kept out of the console logs."""

import asyncio
import inspect
import json
import logging
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from src.graph.graph_state import WorkflowState


logger = logging.getLogger(__name__)

# One file per stage and job; each write replaces the previous one.
RESEARCH_STATE_FILE = "research_state.json"
FINAL_STATE_FILE = "final_state.json"


def _state_value_to_json(value):
    """Convert values json.dumps cannot handle; called again for nested values."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    return str(value)  # datetimes, Paths, anything else


def dump_workflow_state(state: WorkflowState, path: Path) -> None:
    """Write the state as indented JSON, replacing any earlier dump at that path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(state), indent=2, ensure_ascii=False, default=_state_value_to_json),
        encoding="utf-8",
    )


async def run_node_and_dump_state(
    state: WorkflowState,
    *,
    node: Callable,
    directory: Path,
    file_name: str,
) -> dict:
    """
    Run a graph node, then dump the state as it will be after this node's update.

    State fields have no reducers, so merging the update over the input state
    gives the next graph state. A failed dump is logged and never fails the job.
    """
    update = node(state)
    if inspect.isawaitable(update):
        update = await update

    path = directory / state["job_id"] / file_name
    try:
        await asyncio.to_thread(dump_workflow_state, {**state, **update}, path)
    except (OSError, TypeError, ValueError) as exc:
        logger.warning(
            "Workflow state dump failed.",
            extra={"event": "state_dump_failed", "job_id": state["job_id"], "data": {"file": file_name, "error": str(exc)}},
        )
    else:
        print(f"\nworkflow state dumped to {path}")
    return update
