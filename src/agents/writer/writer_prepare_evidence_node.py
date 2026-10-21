"""Deterministic evidence preparation for the Writer Agent."""

import logging

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


def _select_writer_evidence(
    *,
    subquestions: list[SubQuestion],
    evidence: list[Evidence],
) -> list[Evidence]:
    """
    Supplies every accepted Evidence item to the Writer.

    Items are grouped in research-plan order and sorted strongest first within
    each subquestion, so related passages stay together in the prompt instead of
    being scattered across a long context.
    """

    grouped_evidence = _group_evidence_by_subquestion(
        subquestions=subquestions,
        evidence=evidence,
    )
    return [
        evidence_item
        for subquestion in subquestions
        for evidence_item in grouped_evidence[subquestion.id]
    ]


def writer_prepare_evidence_node(
    state: WorkflowState,
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

    # Per-subquestion counts for progress details; contents stay in research state.
    evidence_summary = [
        {
            "question": subquestion.question,
            "evidence_count": sum(item.subquestion_id == subquestion.id for item in selected_evidence),
            "source_count": len({
                item.source_id for item in selected_evidence if item.subquestion_id == subquestion.id
            }),
        }
        for subquestion in subquestions
    ]

    return {
        "writer": {
            **state["writer"],
            "evidence_ids": selected_evidence_ids,
            "source_stats": source_stats,
            "evidence_summary": evidence_summary,
        }
    }


