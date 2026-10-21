"""Writer Critic node: reviews the first draft and may request one revision."""

import logging

from src.graph.graph_state import WorkflowState
from src.llm import LLMCallMetadata, LLMRole, LLMService, LLMServiceError
from src.models import CriticReview, ErrorType, ResearchError, ResearchSource
from src.prompts.critic_prompt import build_critic_messages


logger = logging.getLogger(__name__)


async def writer_critic_node(
    state: WorkflowState,
    *,
    llm_service: LLMService,
) -> dict:
    """
    Reviews the draft for grounding, completeness, and clarity.

    The critic never sends the job back to research: a "revise" verdict leads to
    one writer revision with the same evidence. A critic failure approves the
    draft so citation checking still runs.
    """

    log_context = {
        "job_id": state["job_id"],
        "research_session_id": state["research_session_id"],
        "node": "writer_critic",
        "iteration": state["research"]["iteration_count"],
    }
    logger.info("Writer critic review started.", extra={**log_context, "event": "writer_critic_started"})

    draft_report = state["writer"]["draft_report"]
    evidence_by_id = {item.id: item for item in state["research"]["evidence"]}
    evidence = [
        evidence_by_id[identifier]
        for identifier in dict.fromkeys(state["writer"]["evidence_ids"])
        if identifier in evidence_by_id
    ]

    # A deterministic no-evidence report has nothing to review.
    if draft_report is None or not draft_report.findings or not evidence:
        review = CriticReview(verdict="approve")
        print("\ncritic skipped: no findings or evidence to review")
        return {"writer": {**state["writer"], "critic_review": review}}

    messages = build_critic_messages(
        original_question=state["question"],
        draft_report=draft_report,
        evidence=evidence,
    )
    print("\ncritic prompt:", messages)

    errors = list(state["errors"])
    try:
        review = await llm_service.invoke_structured(
            role=LLMRole.CRITIC,
            messages=messages,
            output_schema=CriticReview,
            metadata=LLMCallMetadata(
                **log_context,
                contains_private_data=ResearchSource.DOCUMENTS in state["allowed_sources"],
            ),
        )
    except LLMServiceError as exc:
        logger.exception(
            "Writer critic review failed; continuing with the draft.",
            extra={**log_context, "event": "writer_critic_failed", "error_type": ErrorType.SYNTHESIS.value},
        )
        errors.append(ResearchError(
            error_type=ErrorType.SYNTHESIS,
            component="writer_critic",
            message=f"Critic review failed; the draft went to citation checking unreviewed: {exc}",
            recoverable=True,
        ))
        review = CriticReview(verdict="approve")

    # A "revise" verdict without any actionable issue gives the writer nothing to fix.
    if review.verdict == "revise" and not review.issues:
        review = CriticReview(verdict="approve")

    print("\ncritic review:", review)
    logger.info(
        "Writer critic review completed.",
        extra={
            **log_context,
            "event": "writer_critic_completed",
            "data": {
                "verdict": review.verdict,
                "issue_count": len(review.issues),
                "issues": [
                    {
                        "finding_index": issue.finding_index,
                        "issue_type": issue.issue_type,
                        "instruction": issue.instruction[:300],
                    }
                    for issue in review.issues
                ],
            },
        },
    )
    return {
        "writer": {**state["writer"], "critic_review": review},
        "errors": errors,
    }
