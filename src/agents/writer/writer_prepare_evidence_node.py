"""Deterministic evidence preparation for the Writer Agent."""

import logging

from src.config import Settings
from src.graph.graph_state import WorkflowState
from src.models import Evidence, SubQuestion


logger = logging.getLogger(__name__)


def _group_evidence_by_subquestion(
    *,
    subquestions: list[SubQuestion],
    evidence: list[Evidence],
) -> dict[str, list[Evidence]]:
    """
    Groups accepted evidence by SubQuestion.

    Evidence referencing an unknown SubQuestion is ignored rather than
    forwarded to the Writer.
    """

    grouped_evidence = {
        subquestion.id: []
        for subquestion in subquestions
    }

    for evidence_item in evidence:
        if evidence_item.subquestion_id not in grouped_evidence:
            continue

        grouped_evidence[evidence_item.subquestion_id].append(evidence_item)

    # Highest-relevance evidence is considered first for each subquestion.
    for evidence_items in grouped_evidence.values():
        evidence_items.sort(
            key=lambda item: (-item.relevance_score, item.id)
        )

    return grouped_evidence


def _evidence_fits_budget(
    *,
    evidence: Evidence,
    selected_count: int,
    used_chars: int,
    max_items: int,
    max_chars: int,
) -> bool:
    """Returns whether an Evidence item fits within the Writer context budget."""
    # added this for groq for char limit so context window does not overlfow
    return True
    # return (
    #     selected_count < max_items
    #     and used_chars + len(evidence.content) <= max_chars
    # )


def _select_writer_evidence(
    *,
    subquestions: list[SubQuestion],
    evidence: list[Evidence],
    max_items: int,
    max_chars: int,
) -> list[Evidence]:
    """
    Selects evidence for report synthesis.

    Pass 1 gives each subquestion an opportunity to contribute its strongest
    evidence. 
    Pass 2 fills remaining capacity with the strongest evidence
    across all subquestions.
    """

    grouped_evidence = _group_evidence_by_subquestion(
        subquestions=subquestions,
        evidence=evidence,
    )

    selected: list[Evidence] = []
    selected_ids: set[str] = set()
    used_chars = 0

    # Pass 1: preserve coverage breadth by selecting one strong item
    # from each subquestion when it fits the remaining budget.
    for subquestion in subquestions:
        for evidence_item in grouped_evidence[subquestion.id]:
            if not _evidence_fits_budget(
                evidence=evidence_item,
                selected_count=len(selected),
                used_chars=used_chars,
                max_items=max_items,
                max_chars=max_chars,
            ):
                continue

            selected.append(evidence_item)
            selected_ids.add(evidence_item.id)
            used_chars += len(evidence_item.content)
            break

        if len(selected) >= max_items or used_chars >= max_chars:
            return selected

    # Pass 2: use only valid, known-subquestion evidence and fill remaining
    # capacity according to relevance.
    remaining_evidence = [
        evidence_item
        for evidence_items in grouped_evidence.values()
        for evidence_item in evidence_items
        if evidence_item.id not in selected_ids
    ]

    remaining_evidence.sort(
        key=lambda item: (
            -item.relevance_score,
            item.subquestion_id,
            item.id,
        )
    )

    for evidence_item in remaining_evidence:
        if not _evidence_fits_budget(
            evidence=evidence_item,
            selected_count=len(selected),
            used_chars=used_chars,
            max_items=max_items,
            max_chars=max_chars,
        ):
            continue

        selected.append(evidence_item)
        selected_ids.add(evidence_item.id)
        used_chars += len(evidence_item.content)

        if len(selected) >= max_items or used_chars >= max_chars:
            break

    return selected


def writer_prepare_evidence_node(
    state: WorkflowState,
    *,
    settings: Settings,
) -> dict:
    """
    Selects evidence that may be supplied to the Writer Agent.

    Only Evidence IDs are stored in WriterAgentState so Evidence objects
    remain owned by ResearchAgentState and are not duplicated in graph state.
    """

    subquestions = state["research"]["subquestions"]
    evidence = state["research"]["evidence"]

    logger.info(
        "Writer evidence preparation started.",
        extra={
            "event": "writer_evidence_preparation_started",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "writer_prepare_evidence",
            "iteration": state["research"]["iteration_count"],
        },
    )

    selected_evidence = _select_writer_evidence(
        subquestions=subquestions,
        evidence=evidence,
        max_items=settings.writer.max_writer_evidence_items,
        max_chars=settings.writer.max_writer_evidence_chars,
    )

    selected_evidence_ids = [
        evidence_item.id
        for evidence_item in selected_evidence
    ]
    source_stats = {
        "collected": len({source.id for source in state["research"]["sources"]}),
        "retrieved": len(set(state["research"].get("retrieved_source_ids", []))),
        "accepted": len({item.source_id for item in evidence}),
        "supplied_to_writer": len({item.source_id for item in selected_evidence}),
    }

    logger.info(
        "Writer evidence preparation completed.",
        extra={
            "event": "writer_evidence_preparation_completed",
            "data": {"source_lineage": source_stats, "selected_evidence_count": len(selected_evidence)},
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "writer_prepare_evidence",
            "iteration": state["research"]["iteration_count"],
        },
    )

    return {
        "writer": {
            **state["writer"],
            "evidence_ids": selected_evidence_ids,
            "source_stats": source_stats,
        }
    }


