"""Pre-extraction relevance gate for search candidates."""

from dataclasses import dataclass
from itertools import zip_longest

from langchain_core.embeddings import Embeddings


@dataclass(frozen=True)
class GateCandidate:
    """One deduplicated search candidate, before its page is extracted."""

    key: str
    text: str
    # Subquestions whose queries discovered this candidate. Empty means the
    # candidate was carried over and is matched against every active subquestion.
    subquestion_ids: tuple[str, ...]
    # Search phrases that found the candidate. Short, concrete phrases match
    # specific page titles better than an abstract subquestion does.
    query_texts: tuple[str, ...] = ()


@dataclass(frozen=True)
class GateScore:
    """Gate decision for one candidate."""

    key: str
    subquestion_id: str
    score: float
    decision: str  # "passed", "kept_min", or "rejected"


class SearchRelevanceGate:
    """
    Scores search candidates by meaning before any page is extracted.

    The candidate's title + snippet is compared with its subquestion and with the
    search phrases that found it, using the shared embedding model. Every provider
    is judged on one scale, and off-topic results never reach extraction,
    chunking, or the retrieval index.
    """

    def __init__(self, embeddings: Embeddings, threshold: float = 0.7, min_keep: int = 2):
        self.embeddings = embeddings
        self.threshold = threshold
        self.min_keep = min_keep

    def evaluate(
        self,
        *,
        subquestions: dict[str, str],
        candidates: list[GateCandidate],
    ) -> list[GateScore]:
        """Score candidates against their subquestion and apply threshold + minimum keep."""
        if not candidates or not subquestions:
            return []

        subquestion_ids = list(subquestions)
        # Normalized embeddings make the dot product equal to cosine similarity.
        # Subquestions use the query-side prompt; candidates are embedded as passages.
        query_vectors = {
            subquestion_id: self.embeddings.embed_query(subquestions[subquestion_id])
            for subquestion_id in subquestion_ids
        }
        phrases = list(dict.fromkeys(text for item in candidates for text in item.query_texts))
        phrase_vectors = {text: self.embeddings.embed_query(text) for text in phrases}
        candidate_vectors = self.embeddings.embed_documents([item.text for item in candidates])

        scores: list[GateScore] = []
        for candidate, vector in zip(candidates, candidate_vectors):
            targets = [item for item in candidate.subquestion_ids if item in query_vectors] or subquestion_ids
            best_id, best_score = max(
                ((target, _dot(query_vectors[target], vector)) for target in targets),
                key=lambda pair: pair[1],
            )
            # A page is relevant if it matches its subquestion or the phrase that found it.
            best_score = max([best_score, *(_dot(phrase_vectors[text], vector) for text in candidate.query_texts)])
            scores.append(GateScore(
                key=candidate.key,
                subquestion_id=best_id,
                score=round(best_score, 4),
                decision="passed" if best_score >= self.threshold else "rejected",
            ))

        # Never leave a subquestion without pages: keep its best few below the gate.
        final: list[GateScore] = []
        by_subquestion = _group_by_subquestion(scores)
        for subquestion_id in subquestion_ids:
            ranked = by_subquestion.get(subquestion_id, [])
            missing = max(0, self.min_keep - sum(item.decision == "passed" for item in ranked))
            for item in ranked:
                if item.decision == "rejected" and missing > 0:
                    item = GateScore(item.key, item.subquestion_id, item.score, "kept_min")
                    missing -= 1
                final.append(item)
        return final


def fair_order(scores: list[GateScore], subquestion_ids: list[str]) -> list[GateScore]:
    """Order admitted candidates best-first within each subquestion, alternating subquestions."""
    admitted = [item for item in scores if item.decision != "rejected"]
    groups = _group_by_subquestion(admitted)
    order = [*subquestion_ids, *(item for item in groups if item not in subquestion_ids)]
    return [
        item
        for row in zip_longest(*(groups.get(subquestion_id, []) for subquestion_id in order))
        for item in row
        if item is not None
    ]


def _group_by_subquestion(scores: list[GateScore]) -> dict[str, list[GateScore]]:
    groups: dict[str, list[GateScore]] = {}
    for item in sorted(scores, key=lambda value: -value.score):
        groups.setdefault(item.subquestion_id, []).append(item)
    return groups


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
