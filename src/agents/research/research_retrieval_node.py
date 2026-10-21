"""Retrieve web and private-document context for the current research round."""

import asyncio
import json
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
        for subquestion in (
            state["research"].get("active_subquestions") or state["research"]["subquestions"]
        )
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
) -> list[RetrievalDocument]:
    # Follow-up intents can discover answers in earlier collected sources too.
    # This remains a job-local public corpus, separate from private documents.
    iteration_sources = [
        source
        for source in state["research"]["sources"]
        if source.source_type == SourceType.WEB
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

    logger.info(
        "Research retrieval started.",
        extra={
            "event": "research_retrieval_started",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "research_retrieval",
            "iteration": state["research"]["iteration_count"] + 1,
            "allowed_sources": state["allowed_sources"]
        },
    )

    iteration_queries = _iteration_queries(state)
    requests = _retrieval_requests(state, iteration_queries)
    allowed_sources = state["allowed_sources"]
    web_results: list[RetrievalResult] = []
    document_results: list[RetrievalResult] = []

    resources = config.get("configurable", {}).get("job_resources")
    if not isinstance(resources, ResearchJobResources):
        raise RuntimeError("Research job resources are unavailable.")

    if ResearchSource.WEB in allowed_sources:
        documents = _web_documents(state)
        if documents:
            # One job-owned web session grows across rounds: only sources added
            # since the last round are chunked and embedded. ResearchService
            # clears it with the other job resources on every terminal path.
            if resources.web_retrieval_session is None:
                resources.web_retrieval_session = retrieval_session_factory.create_session()
            web_session = resources.web_retrieval_session
            new_document_count = await asyncio.to_thread(web_session.add, documents)
            logger.info(
                "Web retrieval index updated.",
                extra={
                    "event": "web_retrieval_index_updated",
                    "job_id": state["job_id"],
                    "research_session_id": state["research_session_id"],
                    "node": "research_retrieval",
                    "iteration": state["research"]["iteration_count"] + 1,
                    "data": {
                        "new_document_count": new_document_count,
                        "total_document_count": len(documents),
                    },
                },
            )
            web_results = await _retrieve(web_session, requests)

    if ResearchSource.DOCUMENTS in allowed_sources:
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
            "document_count": len(document_results),
        },
    )

    print("\nretrieval doc source result: ", len(sources))
    # print(
    #     json.dumps(
    #         {
    #             "sources": [
    #                 {
    #                     **source.model_dump(
    #                         mode="json",
    #                         include={
    #                             "id",
    #                             "source_type",
    #                             "provider",
    #                             "canonical_id",
    #                             "search_result_ids",
    #                             "title",
    #                             "uri",
    #                             "retrieved_at",
    #                             "provider_metadata",
    #                         },
    #                     ),
    #                     "content_length": len(source.content),
    #                 }
    #                 for source in sources
    #             ],
    #             "total_content_length": sum(
    #                 len(source.content) for source in sources
    #             ),
    #         },
    #         indent=2,
    #         ensure_ascii=False,
    #     )
    # )

    return {
        "research": {
            **state["research"],
            "retrieved_source_ids": sorted(set(state["research"].get("retrieved_source_ids", [])) | {
                item.chunk.metadata.get("private_document_id") or item.chunk.document_id
                for result in [*web_results, *document_results] for item in result.chunks
            }),
            "sources": sources,
            "web_retrieval_results": web_results,
            "document_retrieval_results": document_results,
        },
    }
