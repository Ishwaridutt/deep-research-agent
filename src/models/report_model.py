"""Structured writer, citation, and final report models."""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from src.models.source_model import SourceType


class ReportFinding(BaseModel):
    """A citation-checked analysis block in a topic-appropriate report section."""

    section: str = Field(
        default="Detailed Analysis", min_length=1,
        description="Free-form section heading chosen for the research topic; reuse it for related findings.",
    )
    title: str = Field(min_length=1)
    content: str = Field(
        min_length=1,
        description="Detailed evidence-grounded paragraphs, lists, or Markdown tables with inline evidence citations.",
    )
    evidence_ids: list[str] = Field(default_factory=list, description="Supplied evidence IDs supporting this block.")


class FinalReportFinding(BaseModel):
    """A user-facing finding with internal evidence IDs removed."""

    section: str = Field(default="Detailed Analysis", min_length=1)
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)


class DraftReport(BaseModel):
    """A topic-adaptive report whose analysis blocks share one citation pipeline."""

    title: str = Field(default="Deep Research Report", min_length=1, description="Specific, plain-text title naming the research subject.")
    executive_summary: str = Field(
        min_length=1, description="Answer-first summary of supported conclusions and material uncertainty, with evidence citations.",
    )
    findings: list[ReportFinding] = Field(
        default_factory=list,
        description="Ordered analysis blocks grouped by free-form section headings. Choose sections and depth to fit the question and evidence.",
    )

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
    reason: str | None
    unsupported_sentences: list[str] = Field(
        default_factory=list,
        description=(
            "When a finding is only partly supported, the unsupported sentences copied verbatim. "
            "Empty when the finding is fully supported or entirely unsupported."
        ),
    )


class CriticIssue(BaseModel):
    """One problem the critic wants the writer to fix."""

    finding_index: int | None = Field(
        default=None,
        description="Index of the affected finding; null for the executive summary or the whole report.",
    )
    issue_type: Literal["unsupported_claim", "missing_requested_part", "contradiction", "unclear"]
    instruction: str = Field(min_length=1, description="Specific fix for the writer.")


class CriticReview(BaseModel):
    """Critic verdict on a draft report: approve it or request one revision."""

    verdict: Literal["approve", "revise"]
    issues: list[CriticIssue] = Field(default_factory=list)


class SemanticCitationCheckOutput(BaseModel):
    """Structured LLM output used during semantic citation checking."""

    revised_executive_summary: str = Field(
        min_length=1, description="Summary supported by validated findings, with inline evidence-ID citations.",
    )
    finding_results: list[SemanticFindingCheck]


class ReportSource(BaseModel):
    """Source metadata exposed in the final report."""

    source_id: str = Field(min_length=1)
    source_type: SourceType = SourceType.WEB
    title: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    locations: list[dict[str, Any]] = Field(default_factory=list)


class UnverifiedMaterial(BaseModel):
    """Draft content whose claims citation checking could not verify, kept for reader judgment."""

    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    sources: list[ReportSource] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """Validated final report returned by the research application."""

    title: str = Field(default="Deep Research Report", min_length=1)
    executive_summary: str = Field(min_length=1)
    findings: list[FinalReportFinding] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    sources: list[ReportSource] = Field(default_factory=list)
    # Not citation-verified; rendered in a separate, clearly labelled section.
    unverified_material: list[UnverifiedMaterial] = Field(default_factory=list)
    markdown: str = ""

