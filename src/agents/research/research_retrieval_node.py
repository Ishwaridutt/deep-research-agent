"""Research Retrieval node."""

import asyncio
import logging
import json
from dataclasses import asdict

from src.graph.graph_state import WorkflowState
from src.models import RetrievalDocument, RetrievalRequest, RetrievalResult
from src.retrieval import RetrievalSessionFactory


logger = logging.getLogger(__name__)


async def research_retrieval_node(
    state: WorkflowState,
    *,
    retrieval_session_factory: RetrievalSessionFactory,
) -> dict:
    """
    Retrieves relevant source chunks for each subquestion in the current iteration.

    Sources are chunked and indexed once, then the same retrieval session is reused
    for every subquestion participating in the iteration.
    """

    target_iteration = state["research"]["iteration_count"] + 1

    # Identify the queries and SearchResults belonging to the current research iteration.
    iteration_queries = [
        query
        for query in state["research"]["search_queries"]
        if query.iteration == target_iteration
    ]

    query_ids = {query.id for query in iteration_queries}

    search_result_ids = {
        result.id
        for result in state["research"]["search_results"]
        if result.query_id in query_ids
    }

    # Include existing Sources when they were rediscovered during this iteration.
    iteration_sources = [
        source
        for source in state["research"]["sources"]
        if search_result_ids.intersection(source.search_result_ids)
    ]

    logger.info(
        "Research retrieval started.",
        extra={
            "event": "research_retrieval_started",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "research_retrieval",
            "iteration": target_iteration,
            "source_count": len(iteration_sources),
            "query_count": len(iteration_queries),
        },
    )

    retrieval_documents = [
        RetrievalDocument(
            document_id=source.id,
            content=source.content,
            source_type=source.source_type.value,
            title=source.title,
            metadata={
                "url": str(source.uri),
                "search_result_ids": list(source.search_result_ids),
            },
        )
        for source in iteration_sources
        if source.content.strip()
    ]

    # Group search-query text by subquestion so each RetrievalRequest retains search intent.
    queries_by_subquestion: dict[str, list[str]] = {}

    for query in iteration_queries:
        queries_by_subquestion.setdefault(query.subquestion_id, []).append(query.query)

    retrieval_results: list[RetrievalResult] = []

    if retrieval_documents:
        session = retrieval_session_factory.create_session()

        try:
            # Build Chonkie, BM25, and semantic indexes once for the full iteration corpus.
            await asyncio.to_thread(session.build, retrieval_documents)

            for subquestion in state["research"]["subquestions"]:
                search_queries = queries_by_subquestion.get(subquestion.id)

                if not search_queries:
                    continue

                request = RetrievalRequest(
                    subquestion_id=subquestion.id,
                    research_question=state["question"],
                    subquestion=subquestion.question,
                    search_queries=list(dict.fromkeys(search_queries)),
                )

                retrieval_results.append(
                    await asyncio.to_thread(
                        session.retrieve,
                        request,
                    )
                )

                print('\nretrieved data for subquestion: ', subquestion)
                print(retrieval_results)

        finally:
            session.clear()

    logger.info(
        "Research retrieval completed.",
        extra={
            "event": "research_retrieval_completed",
            "job_id": state["job_id"],
            "research_session_id": state["research_session_id"],
            "node": "research_retrieval",
            "iteration": target_iteration,
            "retrieval_result_count": len(retrieval_results),
        },
    )


    print('\nFinal resul of retrieval node: ')
    print(
        json.dumps(
            [asdict(result) for result in retrieval_results],
            indent=2,
            ensure_ascii=False,
        )
    )

    return {
        "research": {
            **state["research"],
            "retrieval_results": retrieval_results,
        },
    }

