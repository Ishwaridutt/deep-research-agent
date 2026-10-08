import re
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from rank_bm25 import BM25Okapi


class RetrievalBM25Retriever(BaseRetriever):
    """LangChain-compatible retriever backed by rank_bm25."""

    documents: list[Document]
    bm25: Any
    k: int = 10

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_documents(cls, documents: list[Document], k: int = 10) -> "RetrievalBM25Retriever":
        """Builds a BM25 retriever from the supplied documents."""

        if k <= 0:
            raise ValueError("k must be greater than 0.")

        # Tokenize the corpus once because the same BM25 index serves all subquestions in the research session.
        tokenized_corpus = [cls._tokenize(document.page_content) for document in documents]
        bm25 = BM25Okapi(tokenized_corpus)

        return cls(documents=list(documents), bm25=bm25, k=k)

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        """Returns documents ordered by BM25 relevance."""

        if not query or not query.strip():
            return []

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Rank by BM25 score and return only the configured number of candidates.
        result_count = min(self.k, len(self.documents))
        ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:result_count]

        return [self.documents[index] for index in ranked_indices]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Creates lightweight lexical tokens while retaining common identifiers."""

        return re.findall(r"\w+(?:[./:-]\w+)*", text.lower())

    