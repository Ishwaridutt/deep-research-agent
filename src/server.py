"""FastAPI application composition root."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from transformers import AutoTokenizer

from src.config import Settings, get_settings
from src.config.logging import configure_logging
from src.documents.document_dependencies import create_document_service
from src.graph.graph_builder import build_research_graph
from src.guardrails.input_classifier import ResearchInputGuard
from src.llm import JevService, create_jev_service, create_llm_service
from src.persistence import ResearchJobStore
from src.retrieval import RetrievalSessionFactory
from src.retrieval.embeddings import RetrievalEmbeddingService
from src.routes import (
    create_research_router,
    document_router,
    health_router,
    index_router,
)
from src.services import ResearchService
from src.services.document_service import DocumentService
from src.tools.cleaning import SourceCleaner
from src.tools.extraction import SourceContentExtractor, create_source_content_extractor
from src.tools.search import SearchProvider, SearchRelevanceGate, create_search_provider


logger = logging.getLogger(__name__)

_SRC_DIRECTORY = Path(__file__).resolve().parent
_STATIC_DIRECTORY = _SRC_DIRECTORY / "static"
_DEFAULT_DB_PATH = _SRC_DIRECTORY.parent / "data" / "research.sqlite3"


class AsyncClosable(Protocol):
    """Contract for long-lived resources requiring async cleanup."""

    async def close(self) -> None:
        """Release resources owned by the object."""
        ...


async def _close_resource(*, name: str, resource: AsyncClosable) -> None:
    """Safely close one long-lived application resource."""

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
            extra={"event": "resource_closed", "node": "server", "resource": name},
        )


def _create_retrieval_session_factory(
    settings: Settings,
    *,
    embedding_service: RetrievalEmbeddingService,
) -> RetrievalSessionFactory:
    """Create shared retrieval dependencies and the per-job session factory."""

    retrieval_settings = settings.retrieval

    # The tokenizer is local; the evidence LLM remains remote.
    context_tokenizer = AutoTokenizer.from_pretrained(
        retrieval_settings.context_tokenizer_model,
    )

    def count_context_tokens(text: str) -> int:
        return len(context_tokenizer.encode(text, add_special_tokens=False))

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
    source_extractor: SourceContentExtractor,
    source_cleaner: SourceCleaner,
    job_store: ResearchJobStore,
    retrieval_session_factory: RetrievalSessionFactory,
    document_service: DocumentService,
    relevance_gate: SearchRelevanceGate,
    jev_service: JevService | None,
) -> ResearchService:
    """Create the complete research application service."""

    llm_service = create_llm_service(settings)
    graph = build_research_graph(
        settings=settings,
        llm_service=llm_service,
        search_provider=search_provider,
        source_extractor=source_extractor,
        source_cleaner=source_cleaner,
        retrieval_session_factory=retrieval_session_factory,
        document_service=document_service,
        relevance_gate=relevance_gate,
        jev_service=jev_service if settings.llm.evidence_use_jev else None,
        state_dump_directory=settings.app.state_dump_directory if settings.app.state_dump_enabled else None,
    )

    return ResearchService(
        graph=graph,
        settings=settings,
        job_store=job_store,
        llm_service=llm_service,
        document_service=document_service,
        input_guard=ResearchInputGuard(
            llm_service,
            classifier_enabled=settings.research_limits.input_classifier_enabled,
            jev_service=jev_service if settings.llm.guardrail_use_jev else None,
        ),
    )


def create_app() -> FastAPI:
    """Create the application and its shared resources."""

    settings = get_settings()
    configure_logging(settings.observability)

    search_provider = create_search_provider(settings)
    source_extractor = create_source_content_extractor(settings)
    source_cleaner = SourceCleaner(
        min_chars=settings.research_limits.min_source_content_chars,
        max_chars=settings.research_limits.max_source_content_chars,
    )
    job_store = ResearchJobStore(Path(os.getenv("RESEARCH_DB_PATH", str(_DEFAULT_DB_PATH))))

    # Web retrieval sessions and private-document indexing share this model and tokenizer.
    embedding_service = RetrievalEmbeddingService(
        model_name=settings.retrieval.embedding_model,
        device=settings.retrieval.embedding_device,
        batch_size=settings.retrieval.embedding_batch_size,
    )
    retrieval_session_factory = _create_retrieval_session_factory(
        settings,
        embedding_service=embedding_service,
    )
    document_service = create_document_service(
        settings=settings,
    )
    # The pre-extraction relevance gate reuses the shared embedding model.
    relevance_gate = SearchRelevanceGate(
        embeddings=embedding_service.embeddings,
        threshold=settings.research_limits.search_relevance_threshold,
        min_keep=settings.research_limits.search_relevance_min_keep,
    )
    # One shared Jev client (None without a TypeSafe key) for the guardrail and evidence scoring.
    jev_service = create_jev_service(settings)
    logger.info(
        "Jev decision model configured." if jev_service else "Jev decision model not configured; using LLM roles.",
        extra={"event": "jev_configured", "node": "server", "data": {
            "enabled": jev_service is not None,
            "guardrail": bool(jev_service) and settings.llm.guardrail_use_jev,
            "evidence": bool(jev_service) and settings.llm.evidence_use_jev,
            "private_data_allowed": bool(jev_service) and jev_service.private_data_allowed,
        }},
    )
    # Build the graph and expensive shared dependencies once per process.
    research_service = _create_research_service(
        settings=settings,
        search_provider=search_provider,
        source_extractor=source_extractor,
        source_cleaner=source_cleaner,
        job_store=job_store,
        retrieval_session_factory=retrieval_session_factory,
        document_service=document_service,
        relevance_gate=relevance_gate,
        jev_service=jev_service,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize persistence and clean up all process-owned resources."""

        await job_store.initialize()
        logger.info(
            "Application started.",
            extra={"event": "application_started", "node": "server"},
        )
        try:
            yield
        finally:
            # Tasks must stop before closing SQLite and shared providers.
            await research_service.shutdown()
            await job_store.close()
            await _close_resource(name="search_provider", resource=search_provider)
            await _close_resource(name="source_extractor", resource=source_extractor)
            if jev_service is not None:
                await _close_resource(name="jev_service", resource=jev_service)
            logger.info(
                "Application stopped.",
                extra={"event": "application_stopped", "node": "server"},
            )

    app = FastAPI(
        title=settings.app.app_name,
        debug=settings.app.debug,
        lifespan=lifespan,
    )
    app.state.document_service = document_service

    app.include_router(index_router)
    app.include_router(health_router)
    app.include_router(create_research_router(research_service=research_service))
    app.include_router(document_router)
    app.mount(
        "/static", StaticFiles(directory=str(_STATIC_DIRECTORY)), name="static",
    )

    @app.middleware("http")
    async def revalidate_frontend_assets(request, call_next):
        """Make browsers revalidate the page and its assets so UI updates load after a restart."""
        response = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response

    return app


app = create_app()
