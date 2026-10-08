"""Logical LLM roles used by research and writing agents."""

from enum import Enum


class LLMRole(str, Enum):
    """
    Identifies the responsibility an LLM is performing.

    Agents depend on logical roles rather than concrete model names.
    Model selection is handled by application configuration.
    """

    PLANNER = "planner"
    QUERY_GENERATOR = "query_generator"
    EVIDENCE_EXTRACTOR = "evidence_extractor"
    COVERAGE_EVALUATOR = "coverage_evaluator"
    WRITER = "writer"
    CITATION_CHECKER = "citation_checker"

    