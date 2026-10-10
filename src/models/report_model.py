"""Structured writer, citation, and final report models."""

from pydantic import BaseModel, Field, field_validator


class ReportFinding(BaseModel):
    """A report finding grounded in specific evidence."""

    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)


class DraftReport(BaseModel):
    """Structured report created by the Writer Agent."""

    executive_summary: str = Field(min_length=1)
    findings: list[ReportFinding] = Field(default_factory=list)

    limitations: list[str] = Field(
        default_factory=list,
        description=(
            "Research limitations. Return an array of strings. "
            "Use an empty array when there are no limitations."
        ),
    )

    @field_validator(
        "limitations",
        mode="before",
    )
    @classmethod
    def normalize_limitations(
        cls,
        value: str | list[str],
    ) -> list[str]:
        """Normalize a single limitation string into a list."""

        if isinstance(value, str):
            value = value.strip()
            return [value] if value else []

        return value


class CitationIssue(BaseModel):
    """Describes one citation or grounding problem."""

    finding_index: int = Field(ge=0)
    evidence_id: str | None = None
    description: str = Field(min_length=1)


class CitationCheckResult(BaseModel):
    """Final result of deterministic and semantic citation validation."""

    valid: bool
    issues: list[CitationIssue] = Field(default_factory=list)


class SemanticFindingCheck(BaseModel):
    """Semantic grounding decision for one report finding."""

    finding_index: int = Field(ge=0)
    supported: bool
    reason: str | None = None


class SemanticCitationCheckOutput(BaseModel):
    """Structured LLM output used during semantic citation checking."""

    revised_executive_summary: str = Field(min_length=1)
    finding_results: list[SemanticFindingCheck] = Field(default_factory=list)


class ReportSource(BaseModel):
    """Source metadata exposed in the final report."""

    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    uri: str = Field(min_length=1)


class ResearchReport(BaseModel):
    """Validated final report returned by the research application."""

    executive_summary: str = Field(min_length=1)
    findings: list[ReportFinding] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    sources: list[ReportSource] = Field(default_factory=list)
    markdown: str = ""

