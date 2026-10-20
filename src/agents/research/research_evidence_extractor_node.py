"""Score retrieved chunks and retain original passages as research evidence."""

import logging
import json
from collections import Counter
from dataclasses import asdict

from src.config import Settings
from src.graph.graph_state import WorkflowState
from src.llm import LLMCallMetadata, LLMRole, LLMService, LLMServiceError
from src.models import (
    ErrorType,
    Evidence,
    EvidenceExtractionOutput,
    ResearchError,
    ResearchSource,
    SearchQuery,
    SourceType,
)
from src.prompts.evidence_extractor_prompt import build_evidence_extractor_messages
from src.services import generate_evidence_id


logger = logging.getLogger(__name__)


async def research_evidence_extractor_node(
    state: WorkflowState,
    *,
    llm_service: LLMService,
    settings: Settings,
) -> dict:
    """Filter chunk relevance without asking the model to rewrite evidence."""

    target_iteration = state["research"]["iteration_count"] + 1
    current_evidence = list(state["research"]["evidence"])
    errors = list(state["errors"])

    max_evidence_items = settings.research_limits.max_evidence_items
    remaining_evidence_capacity = max_evidence_items - len(current_evidence)
    # Counts accompany a bounded, private-safe scoring audit for this round.
    evidence_stats = {
        "iteration": target_iteration,
        "scored_chunk_count": 0,
        "accepted_chunk_count": 0,
        "low_relevance_chunk_count": 0,
        "capacity_skipped_chunk_count": 0,
        "missing_chunk_count": 0,
        "unknown_chunk_count": 0,
        "duplicate_chunk_count": 0,
        "failed_batch_count": 0,
        "total_evidence_count": len(current_evidence),
        "min_evidence_relevance": settings.research_limits.min_evidence_relevance,
        "capacity_reached": remaining_evidence_capacity <= 0,
    }

    log_context = {
        "job_id": state["job_id"],
        "research_session_id": state["research_session_id"],
        "node": "research_evidence_extractor",
        "iteration": target_iteration,
    }
    logger.info(
        "Evidence scoring started.",
        extra={**log_context, "event": "evidence_extraction_started"},
    )

    if remaining_evidence_capacity <= 0:
        logger.warning(
            "Evidence capacity has been reached.",
            extra={
                "event": "evidence_capacity_reached",
                **log_context,
            },
        )

        return {"research": {**state["research"], "evidence_stats": evidence_stats, "evidence_audit": []}}

    subquestions_by_id = {
        subquestion.id: subquestion
        for subquestion in (
            state["research"].get("active_subquestions") or state["research"]["subquestions"]
        )
    }
    original_objective_ids = {item.id for item in state["research"]["subquestions"]}

    existing_evidence_by_key = {
        (evidence.subquestion_id, evidence.chunk_id): evidence
        for evidence in current_evidence
    }
    queries_by_subquestion: dict[str, list[SearchQuery]] = {}
    for query in state["research"]["search_queries"]:
        if query.iteration == target_iteration:
            queries_by_subquestion.setdefault(query.subquestion_id, []).append(query)

    new_evidence: list[Evidence] = []
    evidence_audit: list[dict] = []
    scored_chunk_count = 0
    low_relevance_chunk_count = 0
    min_relevance = settings.research_limits.min_evidence_relevance

    # Retrieval stays source-specific; accepted evidence shares one downstream flow.
    retrieval_results = [
        *state["research"]["web_retrieval_results"],
        *state["research"]["document_retrieval_results"],
    ]

    print('\nretrieval_results----->')
    print(
        json.dumps(
            [asdict(result) for result in retrieval_results],
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    for retrieval_result in retrieval_results:
        subquestion = subquestions_by_id.get(retrieval_result.subquestion_id)

        if subquestion is None:
            continue

        # Coverage and the writer keep assessing the original objective IDs.
        objective_id = subquestion.parent_subquestion_id or subquestion.id
        if objective_id not in original_objective_ids:
            continue

        queries = queries_by_subquestion.get(subquestion.id, [])
        audit_rows = {}
        for retrieved in retrieval_result.chunks:
            chunk = retrieved.chunk
            private = chunk.source_type == SourceType.DOCUMENT.value
            prior = existing_evidence_by_key.get((objective_id, chunk.chunk_id))
            # Persist bounded public previews only. Job progress is not currently
            # owner-authorized, so private contents and file metadata stay out.
            audit_rows[chunk.chunk_id] = {
                "chunk_id": f"Private passage {len(audit_rows) + 1}" if private else chunk.chunk_id,
                "source_type": chunk.source_type,
                "title": "Private document" if private else (chunk.title or chunk.document_id)[:250],
                "uri": "" if private else str(chunk.metadata.get("url") or "")[:1000],
                "preview": "" if private else chunk.content[:800],
                "preview_truncated": not private and len(chunk.content) > 800,
                "relevance_score": prior.relevance_score if prior else None,
                "decision": "previously_accepted" if prior else "not_scored_capacity" if remaining_evidence_capacity <= 0 else "not_scored",
                "queries": [match["query"] for match in retrieval_result.query_matches
                            if chunk.chunk_id in match.get("selected_chunk_ids", [])],
            }
        evidence_audit.append({"question": subquestion.question, "passages": list(audit_rows.values())})
        if remaining_evidence_capacity <= 0:
            continue

        chunks_by_id = {
            retrieved_chunk.chunk.chunk_id: retrieved_chunk.chunk
            for retrieved_chunk in retrieval_result.chunks
            if retrieved_chunk.chunk.content.strip()
            and (objective_id, retrieved_chunk.chunk.chunk_id) not in existing_evidence_by_key
        }
        if not chunks_by_id:
            continue

        messages = build_evidence_extractor_messages(
            original_question=state["question"],
            subquestion=subquestion,
            search_queries=list(dict.fromkeys(query.query for query in queries)),
            chunks=list(chunks_by_id.values()),
        )

        metadata = LLMCallMetadata(
            **log_context,
            contains_private_data=ResearchSource.DOCUMENTS in state["allowed_sources"],
        )

        try:
            print("\nevidence scoring llm prompt: ", messages)
            scoring_output = await llm_service.invoke_structured(
                role=LLMRole.EVIDENCE_EXTRACTOR,
                messages=messages,
                output_schema=EvidenceExtractionOutput,
                metadata=metadata,
            )

            print("\nevidence scoring llm response: ", scoring_output)

        except LLMServiceError as exc:
            for chunk_id in chunks_by_id:
                audit_rows[chunk_id]["decision"] = "scoring_failed"
            evidence_stats["failed_batch_count"] += 1
            print('Evidence scoring error: ', exc)
            logger.exception(
                "Evidence scoring failed.",
                extra={
                    "event": "evidence_extraction_failed",
                    **log_context,
                    "subquestion_id": subquestion.id,
                    "error_type": ErrorType.EVIDENCE.value,
                },
            )

            errors.append(
                ResearchError(
                    error_type=ErrorType.EVIDENCE,
                    component="research_evidence_extractor",
                    message=f"Could not score evidence for subquestion '{subquestion.id}': {exc}",
                    recoverable=True,
                )
            )

            continue

        # Partial output is usable, but never invent a score for missing chunks or
        # choose between ambiguous duplicate judgments. Report contract violations.
        counts = Counter(item.chunk_id for item in scoring_output.evidence)
        invalid_counts = {
            "missing_chunk_count": len(set(chunks_by_id) - counts.keys()),
            "unknown_chunk_count": len(counts.keys() - chunks_by_id.keys()),
            "duplicate_chunk_count": sum(count > 1 for count in counts.values()),
        }
        for key, count in invalid_counts.items():
            evidence_stats[key] += count

        print("\nEvidence score validation:")
        print(
            json.dumps(
                {
                    "expected_chunk_count": len(chunks_by_id),
                    "returned_score_count": len(scoring_output.evidence),
                    "unique_returned_chunk_count": len(counts),
                    **invalid_counts,
                    "missing_chunk_ids": sorted(set(chunks_by_id) - counts.keys()),
                    "unknown_chunk_ids": sorted(counts.keys() - chunks_by_id.keys()),
                    "duplicate_chunk_ids": {
                        chunk_id: count
                        for chunk_id, count in counts.items()
                        if count > 1
                    },
                    "returned_scores": [
                        {
                            "chunk_id": item.chunk_id,
                            "relevance_score": item.relevance_score,
                        }
                        for item in scoring_output.evidence
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        if any(invalid_counts.values()):
            message = f"Incomplete or invalid chunk scores for subquestion '{subquestion.id}': {invalid_counts}"
            logger.warning(message, extra={
                **log_context, 
                "event": "evidence_scores_invalid",
                "subquestion_id": subquestion.id, 
                "data": invalid_counts,
            })
            errors.append(ResearchError(
                error_type=ErrorType.EVIDENCE, component="research_evidence_extractor",
                message=message, recoverable=True,
            ))

        candidates = [
            item for item in scoring_output.evidence
            if item.chunk_id in chunks_by_id and counts[item.chunk_id] == 1
        ]
        for chunk_id in chunks_by_id:
            audit_rows[chunk_id]["decision"] = "missing_score" if not counts[chunk_id] else "duplicate_score"
        for candidate in candidates:
            audit_rows[candidate.chunk_id].update({
                "relevance_score": candidate.relevance_score,
                "decision": "below_threshold" if candidate.relevance_score < min_relevance else "capacity_skipped",
            })
        scored_chunk_count += len(candidates)
        low_relevance_chunk_count += sum(item.relevance_score < min_relevance for item in candidates)

        print('low_relevance_chunk_count: ', low_relevance_chunk_count)
        print('scored_chunk_count: ', scored_chunk_count)
        print("\nValid scoring candidates:")
        print(
            json.dumps(
                {
                    "candidate_count": len(candidates),
                    "candidates": [
                        {
                            "chunk_id": item.chunk_id,
                            "relevance_score": item.relevance_score,
                        }
                        for item in candidates
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )

        # Admit the strongest scored passages first when evidence capacity is tight.
        for candidate in sorted(candidates, key=lambda item: -item.relevance_score):
            if remaining_evidence_capacity <= 0:
                break

            if candidate.relevance_score < min_relevance:
                continue

            chunk = chunks_by_id[candidate.chunk_id]
            audit_rows[candidate.chunk_id]["decision"] = "accepted"

            new_evidence.append(
                Evidence(
                    id=generate_evidence_id(),
                    subquestion_id=objective_id,
                    source_id=(
                        chunk.metadata.get("private_document_id")
                        or chunk.document_id
                    ),
                    chunk_id=chunk.chunk_id,
                    source_type=SourceType(chunk.source_type),
                    provenance={
                        **chunk.metadata,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "title": chunk.title,
                        "section": chunk.section,
                        "research_subquestion_id": subquestion.id,
                        "research_iteration": target_iteration,
                        "research_query_ids": [query.id for query in queries],
                    },
                    content=chunk.content,
                    relevance_score=candidate.relevance_score,
                )
            )

            existing_evidence_by_key[(objective_id, chunk.chunk_id)] = new_evidence[-1]

            remaining_evidence_capacity -= 1

    evidence_stats.update({
        "scored_chunk_count": scored_chunk_count,
        "accepted_chunk_count": len(new_evidence),
        "low_relevance_chunk_count": low_relevance_chunk_count,
        "capacity_skipped_chunk_count": scored_chunk_count - low_relevance_chunk_count - len(new_evidence),
        "capacity_reached": remaining_evidence_capacity <= 0,
        "total_evidence_count": len(current_evidence) + len(new_evidence),
    })
    logger.info(
        "Evidence scoring completed.",
        extra={
            "event": "evidence_extraction_completed",
            **log_context,
            "data": {
                "scored_chunk_count": scored_chunk_count,
                "low_relevance_chunk_count": low_relevance_chunk_count,
                "accepted_chunk_count": len(new_evidence),
                "min_evidence_relevance": min_relevance,
            },
        },
    )

    return {
        "research": {
            **state["research"],
            "evidence_stats": evidence_stats,
            "evidence_audit": evidence_audit,
            "evidence": [
                *current_evidence,
                *new_evidence,
            ],
        },
        "errors": errors,
    }

