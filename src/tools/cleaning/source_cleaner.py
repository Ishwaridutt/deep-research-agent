"""Deterministic text cleanup before public content enters retrieval."""

import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Literal


ContentFormat = Literal["text", "markdown", "html"]


@dataclass(frozen=True, slots=True)
class CleanedSourceContent:
    content: str
    original_chars: int
    cleaned_chars: int
    truncated: bool
    version: str


class SourceCleaningError(ValueError):
    """Raised when extracted content is unusable after cleaning."""


class _HTMLTextExtractor(HTMLParser):
    _IGNORED = {"script", "style", "nav", "footer", "aside", "form", "noscript", "svg"}
    _BLOCKS = {"article", "blockquote", "br", "div", "hr", "main", "p", "pre", "section", "tr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag in self._IGNORED:
            self.ignored_depth += 1
        elif self.ignored_depth == 0:
            if tag in {f"h{level}" for level in range(1, 7)}:
                self.parts.append(f"\n{'#' * int(tag[1])} ")
            elif tag == "li":
                self.parts.append("\n- ")
            elif tag in {"td", "th"}:
                self.parts.append(" | ")
            elif tag in self._BLOCKS:
                self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._IGNORED and self.ignored_depth:
            self.ignored_depth -= 1
        elif self.ignored_depth == 0 and (
            tag in self._BLOCKS or tag == "li" or tag.startswith("h")
        ):
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self.ignored_depth == 0:
            self.parts.append(data)


class SourceCleaner:
    """Normalize extracted public content without using an LLM."""

    VERSION = "deterministic-v1"
    _HTML_PATTERN = re.compile(
        r"<(?:html|body|article|main|section|div|p|h[1-6]|script|style|nav|table)\b",
        re.IGNORECASE,
    )

    def __init__(self, *, min_chars: int, max_chars: int) -> None:
        if min_chars < 1 or max_chars < min_chars:
            raise ValueError("Source cleaning limits are invalid.")
        self.min_chars = min_chars
        self.max_chars = max_chars

    def clean(
        self,
        content: str,
        *,
        content_format: ContentFormat = "text",
    ) -> CleanedSourceContent:
        original_chars = len(content)
        text = unicodedata.normalize("NFKC", content).replace("\r\n", "\n").replace("\r", "\n")
        text = "".join(
            character
            for character in text
            if character in "\n\t" or not unicodedata.category(character).startswith("C")
        )

        if content_format == "html" or self._HTML_PATTERN.search(text):
            parser = _HTMLTextExtractor()
            parser.feed(text)
            parser.close()
            text = "".join(parser.parts)

        # Join words split only by PDF-style line-break hyphenation.
        text = re.sub(r"(?<=\w)-\n(?=[a-z])", "", text)
        text = self._normalize_lines(text)

        truncated = len(text) > self.max_chars
        if truncated:
            limited = text[:self.max_chars]
            text = limited.rsplit("\n", 1)[0].rstrip() or limited.rstrip()

        if len(text) < self.min_chars:
            raise SourceCleaningError(
                f"Cleaned source contains fewer than {self.min_chars} characters."
            )

        return CleanedSourceContent(
            content=text,
            original_chars=original_chars,
            cleaned_chars=len(text),
            truncated=truncated,
            version=self.VERSION,
        )

    @staticmethod
    def _normalize_lines(text: str) -> str:
        lines: list[str] = []
        previous_line: str | None = None
        blank = False

        for raw_line in text.splitlines():
            line = re.sub(r"[ \t]+", " ", raw_line).strip()
            if not line:
                if lines and not blank:
                    lines.append("")
                blank = True
                continue
            if line == previous_line:
                continue
            lines.append(line)
            previous_line = line
            blank = False

        return "\n".join(lines).strip()
