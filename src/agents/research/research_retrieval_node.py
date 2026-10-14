"""Retrieve web and private-document context for the current research round."""

import asyncio
import logging
from datetime import UTC, datetime

from langchain_core.runnables import RunnableConfig

from src.graph.graph_state import WorkflowState
from src.models import (
    ResearchSource,
    RetrievalDocument,
    RetrievalRequest,
    RetrievalResult,
    SearchQuery,
    Source,
    SourceProvider,
    SourceType,
)
from src.retrieval import RetrievalSession, RetrievalSessionFactory
from src.services.research_job_resources import ResearchJobResources


logger = logging.getLogger(__name__)


def _iteration_queries(state: WorkflowState) -> list[SearchQuery]:
    target_iteration = state["research"]["iteration_count"] + 1
    return [
        query
        for query in state["research"]["search_queries"]
        if query.iteration == target_iteration
    ]


def _retrieval_requests(
    state: WorkflowState,
    iteration_queries: list[SearchQuery],
) -> list[RetrievalRequest]:
    queries_by_subquestion: dict[str, list[str]] = {}
    for query in iteration_queries:
        queries_by_subquestion.setdefault(query.subquestion_id, []).append(
            query.query
        )

    return [
        RetrievalRequest(
            subquestion_id=subquestion.id,
            research_question=state["question"],
            subquestion=subquestion.question,
            search_queries=list(dict.fromkeys(search_queries)),
        )
        for subquestion in state["research"]["subquestions"]
        if (search_queries := queries_by_subquestion.get(subquestion.id))
    ]


async def _retrieve(
    session: RetrievalSession,
    requests: list[RetrievalRequest],
) -> list[RetrievalResult]:
    return [
        await asyncio.to_thread(session.retrieve, request)
        for request in requests
    ]


def _web_documents(
    state: WorkflowState,
    iteration_queries: list[SearchQuery],
) -> list[RetrievalDocument]:
    query_ids = {query.id for query in iteration_queries}
    result_ids = {
        result.id
        for result in state["research"]["search_results"]
        if result.query_id in query_ids
    }
    iteration_sources = [
        source
        for source in state["research"]["sources"]
        if source.source_type == SourceType.WEB
        and result_ids.intersection(source.search_result_ids)
    ]

    return [
        RetrievalDocument(
            document_id=source.id,
            content=source.content,
            source_type=source.source_type.value,
            title=source.title,
            metadata={
                "url": str(source.uri),
                "provider": source.provider.value,
                "canonical_id": source.canonical_id,
                "provider_metadata": dict(source.provider_metadata),
                "search_result_ids": list(source.search_result_ids),
            },
        )
        for source in iteration_sources
        if source.content.strip()
    ]


def _with_document_sources(
    sources: list[Source],
    results: list[RetrievalResult],
) -> list[Source]:
    """Register citation metadata without copying complete uploaded files."""

    merged = list(sources)
    known_ids = {source.id for source in merged}

    for result in results:
        for retrieved in result.chunks:
            chunk = retrieved.chunk
            document_id = chunk.metadata.get("private_document_id")
            if not isinstance(document_id, str) or document_id in known_ids:
                continue

            merged.append(
                Source(
                    id=document_id,
                    source_type=SourceType.DOCUMENT,
                    provider=SourceProvider.PRIVATE_DOCUMENT,
                    canonical_id=f"document:{document_id}",
                    title=chunk.metadata.get("filename") or chunk.title or document_id,
                    uri=f"document://{document_id}",
                    content=chunk.content,
                    retrieved_at=datetime.now(UTC),
                )
            )
            known_ids.add(document_id)

    return merged


async def research_retrieval_node(
    state: WorkflowState,
    config: RunnableConfig,
    *,
    retrieval_session_factory: RetrievalSessionFactory,
) -> dict:
    """Retrieve selected source types while keeping their results separate."""

    iteration_queries = _iteration_queries(state)
    requests = _retrieval_requests(state, iteration_queries)
    allowed_sources = state["allowed_sources"]
    web_results: list[RetrievalResult] = []
    document_results: list[RetrievalResult] = []

    if ResearchSource.WEB in allowed_sources:
        documents = _web_documents(state, iteration_queries)
        if documents:
            web_session = retrieval_session_factory.create_session()
            try:
                await asyncio.to_thread(web_session.build, documents)
                web_results = await _retrieve(web_session, requests)
            finally:
                web_session.clear()

    if ResearchSource.DOCUMENTS in allowed_sources:
        resources = config.get("configurable", {}).get("job_resources")
        if not isinstance(resources, ResearchJobResources):
            raise RuntimeError("Research job resources are unavailable.")

        document_session = resources.document_retrieval_session
        if document_session is None:
            raise RuntimeError("Private-document retrieval session is unavailable.")

        document_results = await _retrieve(document_session, requests)

    sources = _with_document_sources(
        state["research"]["sources"],
        document_results,
    )
    logger.info(
        "Research retrieval completed.",
        extra={
            "event": "research_retrieval_completed",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "research_retrieval",
            "iteration": state["research"]["iteration_count"] + 1,
            "web_result_count": len(web_results),
            "document_result_count": len(document_results),
        },
    )

    return {
        "research": {
            **state["research"],
            "sources": sources,
            "web_retrieval_results": web_results,
            "document_retrieval_results": document_results,
        },
    }
