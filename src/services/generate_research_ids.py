"""Helpers for generating application-owned research identifiers."""

from uuid import uuid4


def _generate_id(prefix: str) -> str:
    """
    Generates a unique application identifier with a readable prefix.

    The UUID is shortened for readability while retaining enough entropy
    for the Phase 1 application.
    """

    unique_part = uuid4().hex[:16]

    return f"{prefix}-{unique_part}"


def generate_job_id() -> str:
    """
    Generates an identifier for one execution of the research workflow.
    """

    return _generate_id("JOB")


def generate_research_session_id() -> str:
    """
    Generates an identifier for a long-lived research conversation/session.
    """

    return _generate_id("RS")


def generate_subquestion_id() -> str:
    """
    Generates an identifier for a planned research subquestion.
    """

    return _generate_id("SQ")


def generate_search_query_id() -> str:
    """
    Generates an identifier for a generated web search query.
    """

    return _generate_id("Q")


def generate_search_result_id() -> str:
    """
    Generates an identifier for a normalized search result.
    """

    return _generate_id("SR")


def generate_source_id() -> str:
    """
    Generates an identifier for an extracted and normalized research source.
    """

    return _generate_id("SRC")


def generate_evidence_id() -> str:
    """
    Generates an identifier for one accepted piece of research evidence.
    """

    return _generate_id("EV")

