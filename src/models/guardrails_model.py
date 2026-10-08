"""Models describing input and tool guardrail decisions."""

from enum import Enum

from pydantic import BaseModel


class GuardrailDecision(str, Enum):
    """Possible outcomes of a guardrail check."""

    ALLOW = "allow"
    REJECT = "reject"


class GuardrailResult(BaseModel):
    """Structured result returned by an input guardrail."""

    decision: GuardrailDecision

    category: str | None = None

    user_message: str | None = None

    