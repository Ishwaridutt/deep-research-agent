"""Structured errors produced during research execution."""

from enum import Enum

from pydantic import BaseModel, Field


class ErrorType(str, Enum):
    """Categories of errors that may occur during a research job."""

    VALIDATION = "validation"
    GUARDRAIL = "guardrail"

    LLM = "llm"

    SEARCH = "search"
    EXTRACTION = "extraction"

    EVIDENCE = "evidence"
    COVERAGE = "coverage"

    SYNTHESIS = "synthesis"
    CITATION = "citation"

    INTERNAL = "internal"


class ResearchError(BaseModel):
    """Structured failure that can safely travel through graph state."""

    error_type: ErrorType

    component: str = Field(
        min_length=1,
    )

    message: str = Field(
        min_length=1,
    )

    recoverable: bool

    