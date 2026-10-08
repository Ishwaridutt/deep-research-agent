"""FastAPI application composition root."""

import logging

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer

from src.config import Settings, get_settings
from src.config.logging import configure_logging
from src.graph.graph_builder import build_research_graph
from src.llm import create_llm_service
from src.retrieval import RetrievalSessionFactory
from src.retrieval.embeddings import RetrievalEmbeddingService
from src.routes import create_research_router, health_router, index_router
from src.services import ResearchService
from src.tools.extraction import WebExtractor, create_web_extractor
from src.tools.search import SearchProvider, create_search_provider


logger = logging.getLogger(__name__)

_SRC_DIRECTORY = Path(__file__).resolve().parent
_STATIC_DIRECTORY = _SRC_DIRECTORY / "static"


class AsyncClosable(Protocol):
    """Contract for long-lived resources requiring async cleanup."""

    async def close(self) -> None:
        """Releases resources owned by the object."""

        ...


async def _close_resource(*, name: str, resource: AsyncClosable) -> None:
    """Safely closes one long-lived application resource."""

    try:
        await resource.close()

    except Exception:
        logger.exception(
            "Application resource cleanup failed.",
            extra={
                "event": "resource_cleanup_failed",
                "node": "server",
                "resource": name,
                "error_type": "internal",
            },
        )

    else:
        logger.info(
            "Application resource closed.",
            extra={
                "event": "resource_closed",
                "node": "server",
                "resource": name,
            },
        )


def _create_retrieval_session_factory(settings: Settings) -> RetrievalSessionFactory:
    """Creates shared retrieval dependencies and the per-job session factory."""

    retrieval_settings = settings.retrieval

    # Load the local embedding model once for the lifetime of the server process.
    embedding_service = RetrievalEmbeddingService(
        model_name=retrieval_settings.embedding_model,
        device=retrieval_settings.embedding_device,
        batch_size=retrieval_settings.embedding_batch_size,
    )

    # Only the tokenizer is loaded here; the evidence LLM itself remains remote.
    context_tokenizer = AutoTokenizer.from_pretrained(
        retrieval_settings.context_tokenizer_model,
    )

    def count_context_tokens(text: str) -> int:
        return len(
            context_tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    return RetrievalSessionFactory(
        embedding_service=embedding_service,
        context_token_counter=count_context_tokens,
        chunk_size=retrieval_settings.chunk_size,
        candidate_k=retrieval_settings.candidate_k,
        top_k=retrieval_settings.top_k,
        semantic_weight=retrieval_settings.semantic_weight,
        lexical_weight=retrieval_settings.lexical_weight,
        rrf_c=retrieval_settings.rrf_c,
        max_context_tokens=retrieval_settings.max_context_tokens,
        max_chunks_per_document=retrieval_settings.max_chunks_per_document,
    )


def _create_research_service(
    *,
    settings: Settings,
    search_provider: SearchProvider,
    web_extractor: WebExtractor,
) -> ResearchService:
    """Creates the complete research application service."""

    llm_service = create_llm_service(settings)
    retrieval_session_factory = _create_retrieval_session_factory(settings)

    graph = build_research_graph(
        settings=settings,
        llm_service=llm_service,
        search_provider=search_provider,
        web_extractor=web_extractor,
        retrieval_session_factory=retrieval_session_factory,
    )

    return ResearchService(
        graph=graph,
        settings=settings,
    )


def create_app() -> FastAPI:
    """Creates and configures the FastAPI application."""

    settings = get_settings()
    configure_logging(settings.observability)

    search_provider = create_search_provider(settings)
    web_extractor = create_web_extractor(settings)

    # Create the graph and shared application dependencies once per server process.
    research_service = _create_research_service(
        settings=settings,
        search_provider=search_provider,
        web_extractor=web_extractor,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Manages process-wide external resources."""

        logger.info(
            "Application started.",
            extra={
                "event": "application_started",
                "node": "server",
            },
        )

        try:
            yield

        finally:
            await _close_resource(
                name="search_provider",
                resource=search_provider,
            )

            await _close_resource(
                name="web_extractor",
                resource=web_extractor,
            )

            logger.info(
                "Application stopped.",
                extra={
                    "event": "application_stopped",
                    "node": "server",
                },
            )

    app = FastAPI(
        title=settings.app.app_name,
        debug=settings.app.debug,
        lifespan=lifespan,
    )

    # Register API and UI routes after shared services are initialized.
    app.include_router(index_router)
    app.include_router(health_router)
    app.include_router(
        create_research_router(
            research_service=research_service,
        )
    )

    app.mount(
        "/static",
        StaticFiles(directory=str(_STATIC_DIRECTORY)),
        name="static",
    )

    return app


app = create_app()

