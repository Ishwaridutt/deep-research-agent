"""Research Planner node."""

import asyncio
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime

from src.config import Settings
from src.graph.graph_state import WorkflowState
from src.llm import (
    LLMCallMetadata,
    LLMRole,
    LLMService,
    LLMServiceError,
)
from src.models import (
    ErrorType,
    FollowupPlannerOutput,
    PlannerOutput,
    ResearchError,
    ResearchSource,
    ResearchStopReason,
    SubQuestion,
)
from src.models.research_quality import ResearchDecision
from src.prompts.research_planner_prompt import build_planner_messages
from src.services import generate_subquestion_id


logger = logging.getLogger(__name__)


def _normalize_text(value: str) -> str:
    """
    Normalizes text for deterministic duplicate detection.
    """

    return " ".join(
        value.strip().lower().split()
    )


def _build_subquestions(
    *,
    planner_output: PlannerOutput | FollowupPlannerOutput,
    max_subquestions: int,
    previous_subquestions: Sequence[SubQuestion] = (),
    allowed_parent_ids: set[str] | None = None,
    iteration: int = 1,
) -> list[SubQuestion]:
    """
    Converts LLM planner output into application-owned SubQuestion objects.

    The application owns IDs and deterministic duplicate filtering.
    """

    subquestions: list[SubQuestion] = []
    seen_questions = {_normalize_text(item.question) for item in previous_subquestions}
    for planned_subquestion in planner_output.subquestions:
        if len(subquestions) >= max_subquestions:
            break
        question = planned_subquestion.question.strip()
        objective = planned_subquestion.objective.strip()
        parent_id = getattr(planned_subquestion, "parent_subquestion_id", None)
        if allowed_parent_ids is not None and parent_id not in allowed_parent_ids:
            continue
        if not question or not objective:
            continue

        normalized_question = _normalize_text(question)
        if normalized_question in seen_questions:
            continue
        seen_questions.add(normalized_question)
        subquestions.append(
            SubQuestion(
                id=generate_subquestion_id(),
                question=question,
                objective=objective,
                parent_subquestion_id=parent_id,
                iteration=iteration,
            )
        )

    return subquestions


def _refinement_context(state: WorkflowState, available_query_slots: int, max_queries_per_subquestion: int) -> dict:
    """Build follow-up planning context from tasks, searches, gaps, and limits."""
    research = state["research"]
    gaps = sorted(state["research_control"].pending_gaps, key=lambda gap: gap.priority, reverse=True)
    history = research.get("planning_history") or research["subquestions"]
    return {
        "original_subquestions": [item.model_dump(mode="json") for item in research["subquestions"]],
        "previous_tasks": [
            {
                **item.model_dump(mode="json"),
                "queries": [query.query for query in research["search_queries"] if query.subquestion_id == item.id],
            }
            for item in history
        ],
        "unresolved_gaps": [gap.model_dump(mode="json") for gap in gaps],
        "available_query_slots": available_query_slots,
        "max_queries_per_subquestion": max_queries_per_subquestion,
    }


def _stop_refinement(state: WorkflowState, reason: ResearchStopReason) -> dict:
    """An exhausted research plan goes to the writer with accumulated evidence intact."""
    control = state["research_control"]
    logger.info("Research refinement stopped.", extra={
        "event": "research_refinement_stopped", "job_id": state["job_id"],
        "iteration": state["research"]["iteration_count"] + 1,
        "data": {"stop_reason": reason.value},
    })
    return {
        "research": {**state["research"], "active_subquestions": [], "stop_reason": reason},
        "research_control": control.model_copy(update={
            "last_decision": ResearchDecision(
                action="write_report", reason=reason.value.replace("_", " "),
                has_unresolved_gaps=bool(control.pending_gaps),
                budget_exhausted=reason in {
                    ResearchStopReason.TIME_BUDGET_EXHAUSTED,
                    ResearchStopReason.QUERY_BUDGET_EXHAUSTED,
                },
            ),
        }),
    }

# graph builder will inject these dependencies.
async def research_planner_node(
    state: WorkflowState,
    *,
    llm_service: LLMService,
    settings: Settings,
) -> dict:
    """
    Decomposes the user's research question into focused subquestions.
    """

    logger.info(
        "Research planning started.",
        extra={
            "event": "research_planning_started",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "research_planner",
        },
    )

    research = state["research"]
    control = state["research_control"]
    is_followup = research["iteration_count"] > 0
    # research time and query limit validation
    # calculates how many queries remain in the job
    remaining_queries = max(0, min(
        settings.research_limits.max_total_queries_per_job,
        control.budget.max_search_queries,
    ) - max(len(research["search_queries"]), control.usage.queries_executed))
    seconds_remaining = max(0.0, control.budget.max_research_seconds - (
        datetime.now(UTC) - control.usage.started_at
    ).total_seconds())
    if seconds_remaining <= 0:
        return _stop_refinement(state, ResearchStopReason.TIME_BUDGET_EXHAUSTED)
    if not remaining_queries:
        return _stop_refinement(state, ResearchStopReason.QUERY_BUDGET_EXHAUSTED)
    if is_followup and not control.pending_gaps:
        return _stop_refinement(state, ResearchStopReason.NO_ACTIONABLE_GAPS)

    # Every selected task needs at least one search from the remaining job budget.
    max_subquestions = min(settings.research_limits.max_subquestions, remaining_queries)
    planner_prompt = build_planner_messages(
        question=state["question"],
        max_subquestions=max_subquestions,
        current_date=datetime.now(UTC).date().isoformat(),
        refinement_context=_refinement_context(
            state, remaining_queries, settings.research_limits.max_queries_per_subquestion,
        ) if is_followup else None,
    )

    print('\nresearch_planner_prompt: ')
    print(planner_prompt)

    metadata = LLMCallMetadata(
        job_id=state["job_id"],
        research_session_id=state["research_session_id"],
        node="research_planner",
        iteration=research["iteration_count"] + 1,
        contains_private_data=ResearchSource.DOCUMENTS in state["allowed_sources"],
    )

    try:
        planner_output = await asyncio.wait_for(
            llm_service.invoke_structured(
                role=LLMRole.PLANNER,
                messages=planner_prompt,
                output_schema=FollowupPlannerOutput if is_followup else PlannerOutput,
                metadata=metadata,
            ),
            timeout=seconds_remaining,
        )
        print("\nplanner output:", planner_output.model_dump_json(indent=2))
      
    except TimeoutError:
        return _stop_refinement(state, ResearchStopReason.TIME_BUDGET_EXHAUSTED)
    except LLMServiceError as exc:
        logger.exception(
            "Research planning failed.",
            extra={
                "event": "research_planning_failed",
                "job_id": state["job_id"],
                "research_session_id": state[
                    "research_session_id"
                ],
                "node": "research_planner",
                "error_type": ErrorType.LLM.value,
            },
        )

        error = ResearchError(
            error_type=ErrorType.LLM,
            component="research_planner",
            message=str(exc),
            recoverable=False,
        )

        return {
            "errors": [
                *state["errors"],
                error,
            ]
        }

    planned_subquestions = _build_subquestions(
        planner_output=planner_output,
        max_subquestions=max_subquestions,
        previous_subquestions=(research.get("planning_history") or research["subquestions"]) if is_followup else (),
        allowed_parent_ids={gap.subquestion_id for gap in control.pending_gaps} & {
            item.id for item in research["subquestions"]
        } if is_followup else None,
        iteration=research["iteration_count"] + 1,
    )

    if not planned_subquestions:
        if is_followup:
            update = _stop_refinement(state, ResearchStopReason.NO_ACTIONABLE_GAPS)
            if planner_output.subquestions:
                logger.warning("Planner returned only invalid or repeated follow-up tasks.", extra={
                    "event": "research_refinement_rejected", "job_id": state["job_id"],
                    "iteration": research["iteration_count"] + 1,
                })
                update["errors"] = [*state["errors"], ResearchError(
                    error_type=ErrorType.VALIDATION, component="research_planner",
                    message="Planner returned no valid new follow-up tasks; reporting available evidence and limitations.",
                    recoverable=True,
                )]
            return update
        logger.error(
            "Research Planner returned no valid subquestions.",
            extra={
                "event": "research_planning_failed",
                "job_id": state["job_id"],
                "research_session_id": state["research_session_id"],
                "node": "research_planner",
                "error_type": (
                    ErrorType.VALIDATION.value
                ),
            },
        )

        error = ResearchError(
            error_type=ErrorType.VALIDATION,
            component="research_planner",
            message=(
                "The Research Planner did not produce "
                "any valid subquestions."
            ),
            recoverable=False,
        )

        return {
            "errors": [
                *state["errors"],
                error,
            ]
        }

    logger.info(
        "Research planning completed.",
        extra={
            "event": "research_planning_completed",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "research_planner",
            "subquestion_count": len(planned_subquestions),
            "iteration": research["iteration_count"] + 1,
            "data": {
                "planning_mode": "refinement" if is_followup else "initial",
                "active_subquestion_count": len(planned_subquestions),
            },
        },
    )
    print("\ngenerated subquestions: ", json.dumps(
        [item.model_dump() for item in planned_subquestions],
        indent=2,
    ))

    return {
        "research": {
            **state["research"],
            # Original objectives are set once in round 1. Follow-ups never replace them:
            # their evidence is filed under the parent objective, and coverage and the
            # writer assess the original objectives.
            "subquestions": research["subquestions"] if is_followup else planned_subquestions,
            # This round's tasks: what the query generator, retrieval, and scorer use.
            "active_subquestions": planned_subquestions,
            "planning_history": [
                *(research.get("planning_history") or research["subquestions"]),
                *planned_subquestions,
            ],
        }
    }


