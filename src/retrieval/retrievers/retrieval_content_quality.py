"""Conservative public-passage noise detection; never rewrite source content."""

import re

from src.models.retrieval_model import RetrievalChunk


_REFERENCE = re.compile(
    r"(?:^|\n)\s*(?:[-*•]\s*)?(?:(?:\[\d+\]|\d+[.)])\s+)?"
    r"[A-Z][a-z’'-]+,?\s+(?:[A-Z]\.?){1,3}(?=[\s,;])"
    r"(?=[^\n]*\b(?:19|20)\d{2}\b)"
    r"(?=[^\n]*(?:doi[:.]|PMID|CrossRef|Google Scholar|\b\d+\s*[:;]\s*\d+))[^\n]*"
)
_LINK_LINE = re.compile(r"^\s*(?:[-*•]\s*)?\[[^\]]+\]\([^\n]+\)\s*$")
_NAVIGATION = {"loading...", "show more", "share", "cite", "download", "add to mendeley"}
_FINDING = re.compile(r"\b(?:found|reported|observed|showed|concluded|confidence interval)\b", re.IGNORECASE)


def is_public_noise(chunk: RetrievalChunk) -> bool:
    """Exclude obvious link menus or citation lists, not substantive prose/tables."""
    if chunk.source_type != "web":
        return False
    text = chunk.content.strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return True
    if all(line.casefold() in _NAVIGATION or _LINK_LINE.fullmatch(line) for line in lines):
        return True
    if _FINDING.search(text):
        return False  # Prefer a false negative over removing a cited finding.
    # Require bibliographic identifiers and high citation density. A References
    # heading or an in-text citation alone is not grounds for discarding a chunk.
    references = list(_REFERENCE.finditer(text))
    reference_chars = sum(len(match.group()) for match in references)
    return bool(references) and reference_chars >= len(text) * (0.6 if len(references) >= 3 else 0.85)
