"""Research Planner node."""

import logging
import json

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
    PlannerOutput,
    ResearchError,
    SubQuestion,
)
from src.prompts.planner_prompt import build_planner_messages
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
    planner_output: PlannerOutput,
    max_subquestions: int,
) -> list[SubQuestion]:
    """
    Converts LLM planner output into application-owned SubQuestion objects.

    The application owns IDs and deterministic duplicate filtering.
    """

    subquestions: list[SubQuestion] = []

    seen_questions: set[str] = set()

    for planned_subquestion in planner_output.subquestions:
        question = (
            planned_subquestion.question.strip()
        )

        objective = (
            planned_subquestion.objective.strip()
        )

        if not question or not objective:
            continue

        normalized_question = (
            _normalize_text(question)
        )

        if normalized_question in seen_questions:
            continue

        seen_questions.add(
            normalized_question
        )

        subquestions.append(
            SubQuestion(
                id=generate_subquestion_id(),
                question=question,
                objective=objective,
            )
        )

        if len(subquestions) >= max_subquestions:
            break

    return subquestions

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

    planner_prompt = build_planner_messages(
        question=state["question"],
        max_subquestions=settings.research_limits.max_subquestions,
    )

    metadata = LLMCallMetadata(
        job_id=state["job_id"],
        research_session_id=state["research_session_id"],
        node="research_planner",
    )

    try:
        planner_output = await llm_service.invoke_structured(
                role=LLMRole.PLANNER,
                messages=planner_prompt,
                output_schema=PlannerOutput,
                metadata=metadata,
            )

        print('\n planner output:', planner_output.model_dump_json(indent=2))
      
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

    subquestions = _build_subquestions(
        planner_output=planner_output,
        max_subquestions=settings.research_limits.max_subquestions,
    )

    if not subquestions:
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
            # "data": subquestions,
        },
    )
    print('\nsubquestions:', json.dumps(
        [item.model_dump() for item in subquestions],
        indent=2,
    ),)
    # pprint.pprint(subquestions)
    # pprint.pprint( 
    #     subquestions,
    #     width=120,
    #     sort_dicts=False,
    #     )

    return {
        "research": {
            **state["research"],
            "subquestions": subquestions,
        }
    }


