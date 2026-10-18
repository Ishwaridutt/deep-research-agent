"""Research Evidence Extractor node."""

import logging
import json
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
    SourceType,
)
from src.prompts.evidence_extractor_prompt import build_evidence_extractor_messages
from src.services import generate_evidence_id


logger = logging.getLogger(__name__)


def _normalize_evidence_text(value: str) -> str:
    """Normalizes evidence text for deterministic duplicate detection."""

    return " ".join(value.strip().lower().split())


async def research_evidence_extractor_node(
    state: WorkflowState,
    *,
    llm_service: LLMService,
    settings: Settings,
) -> dict:
    """Extracts evidence from the retrieved context for the current iteration."""

    target_iteration = state["research"]["iteration_count"] + 1
    current_evidence = list(state["research"]["evidence"])
    errors = list(state["errors"])

    max_evidence_items = settings.research_limits.max_evidence_items
    remaining_evidence_capacity = max_evidence_items - len(current_evidence)

    logger.info(
        "Evidence extraction started.",
        extra={
            "event": "evidence_extraction_started",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "research_evidence_extractor",
            "iteration": target_iteration,
        },
    )

    if remaining_evidence_capacity <= 0:
        logger.warning(
            "Evidence capacity has been reached.",
            extra={
                "event": "evidence_capacity_reached",
                "job_id": state["job_id"],
                "research_session_id": state["research_session_id"],
                "node": "research_evidence_extractor",
                "iteration": target_iteration,
            },
        )

        return {}

    subquestions_by_id = {
        subquestion.id: subquestion
        for subquestion in (
            state["research"].get("active_subquestions") or state["research"]["subquestions"]
        )
    }
    original_objective_ids = {item.id for item in state["research"]["subquestions"]}

    existing_evidence_keys = {
        (
            evidence.subquestion_id,
            _normalize_evidence_text(evidence.content),
        )
        for evidence in current_evidence
    }

    new_evidence: list[Evidence] = []

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
        if remaining_evidence_capacity <= 0:
            break

        if not retrieval_result.context:
            continue

        subquestion = subquestions_by_id.get(retrieval_result.subquestion_id)

        if subquestion is None:
            continue

        # Coverage and the writer keep assessing the original objective IDs.
        objective_id = subquestion.parent_subquestion_id or subquestion.id
        if objective_id not in original_objective_ids:
            continue

        chunks_by_id = {
            retrieved_chunk.chunk.chunk_id: retrieved_chunk.chunk
            for retrieved_chunk in retrieval_result.chunks
        }

        evidence_extraction_prompt = build_evidence_extractor_messages(
            original_question=state["question"],
            subquestion=subquestion,
            retrieval_result=retrieval_result,
        )

        metadata = LLMCallMetadata(
            job_id=state["job_id"],
            research_session_id=state["research_session_id"],
            node="research_evidence_extractor",
            iteration=target_iteration,
            contains_private_data=ResearchSource.DOCUMENTS in state["allowed_sources"],
        )

        try:
            print("\nevidence extraction llm prompt: ", evidence_extraction_prompt)
            extraction_output = await llm_service.invoke_structured(
                role=LLMRole.EVIDENCE_EXTRACTOR,
                messages=evidence_extraction_prompt,
                output_schema=EvidenceExtractionOutput,
                metadata=metadata,
            )

            print("\nevidence extraction llm response: ", extraction_output)

        except LLMServiceError as exc:
            print('Evidence extraction error: ', exec)
            logger.exception(
                "Evidence extraction failed",
                extra={
                    "event": "evidence_extraction_failed",
                    "job_id": state["job_id"],
                    "research_session_id": state["research_session_id"],
                    "node": "research_evidence_extractor",
                    "iteration": target_iteration,
                    "subquestion_id": subquestion.id,
                    "error_type": ErrorType.EVIDENCE.value,
                },
            )

            errors.append(
                ResearchError(
                    error_type=ErrorType.EVIDENCE,
                    component="research_evidence_extractor",
                    message=f"Could not extract evidence for subquestion '{subquestion.id}': {exc}",
                    recoverable=True,
                )
            )

            continue

        # Validate provenance and retain only relevant, unique evidence items.
        for candidate in extraction_output.evidence:
            if remaining_evidence_capacity <= 0:
                break

            content = candidate.content.strip()
            chunk = chunks_by_id.get(candidate.chunk_id)

            if not content or chunk is None:
                continue

            if candidate.relevance_score < settings.research_limits.min_evidence_relevance:
                continue

            evidence_key = (
                objective_id,
                _normalize_evidence_text(content),
            )

            if evidence_key in existing_evidence_keys:
                continue

            existing_evidence_keys.add(evidence_key)

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
                    },
                    content=content,
                    relevance_score=candidate.relevance_score,
                )
            )

            remaining_evidence_capacity -= 1

    logger.info(
        "Evidence extraction completed.",
        extra={
            "event": "evidence_extraction_completed",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "research_evidence_extractor",
            "iteration": target_iteration,
        },
    )

    return {
        "research": {
            **state["research"],
            "evidence": [
                *current_evidence,
                *new_evidence,
            ],
        },
        "errors": errors,
    }

