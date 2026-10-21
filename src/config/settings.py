"""Typed application configuration loaded from environment variables."""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class AppEnvironment(str, Enum):
    """Supported application runtime environments."""

    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class EnvironmentSettings(BaseSettings):
    """Loads common BaseSettings configuration from the project .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


class ApplicationSettings(EnvironmentSettings):
    """Application-level settings."""

    app_name: str = "Deep Research Agent"
    app_env: AppEnvironment = AppEnvironment.DEVELOPMENT
    debug: bool = False

    graph_recursion_limit: int = Field(
        default=50,
        ge=10,
    )


class LLMSettings(EnvironmentSettings):
    """Configuration for provider-independent LLM execution."""

    llm_provider: Literal["openai", "google_genai", "groq"] = "groq"
    llm_provider_order: str = "openai,google_genai,groq"
    llm_private_data_providers: str = "openai,google_genai,groq"

    planner_model: str = Field(min_length=1)
    query_generator_model: str = Field(min_length=1)
    evidence_model: str = Field(min_length=1)
    coverage_model: str = Field(min_length=1)
    writer_model: str = Field(min_length=1)
    citation_model: str = Field(min_length=1)
    # Optional so existing .env files keep working: the critic falls back to the
    # citation model and the input guardrail to the query generator model.
    critic_model: str | None = None
    guardrail_model: str | None = None

    llm_timeout_seconds: int = Field(default=60, ge=1)

    # Provider-level retries for transient transport/API failures.
    llm_max_retries: int = Field(default=2, ge=0)

    llm_structured_output_method: Literal[
        "json_schema",
        "function_calling",
    ] = "json_schema"

    # Total structured-output attempts, including the initial call.
    llm_structured_output_attempts: int = Field(
        default=2,
        ge=1,
        le=3,
    )


class SearchSettings(EnvironmentSettings):
    """Configuration for web search and webpage extraction providers."""

    tavily_api_key: SecretStr = Field(min_length=1)
    brave_api_key: SecretStr | None = None
    firecrawl_api_key: SecretStr | None = None

    search_provider: str = "tavily"
    web_extractor_provider: str = "tavily"

    search_timeout_seconds: int = Field(
        default=20,
        ge=1,
    )

    extraction_timeout_seconds: int = Field(
        default=30,
        ge=1,
    )

    tavily_extract_depth: Literal[
        "basic",
        "advanced",
    ] = "basic"

    tavily_extract_format: Literal[
        "markdown",
        "text",
    ] = "markdown"

    brave_api_url: str = "https://api.search.brave.com/res/v1/web/search"
    brave_api_version: str = "2023-01-01"
    firecrawl_api_url: str = "https://api.firecrawl.dev/v2/scrape"

    provider_max_retries: int = Field(default=1, ge=0)
    provider_retry_delay_seconds: float = Field(default=0.25, ge=0.0)
    provider_circuit_failure_threshold: int = Field(default=2, ge=1)
    provider_circuit_recovery_seconds: float = Field(default=60.0, ge=0.0)

    arxiv_enabled: bool = True
    wikipedia_enabled: bool = True

    arxiv_api_url: str = "https://export.arxiv.org/api/query"
    wikipedia_api_url: str = "https://en.wikipedia.org/w/api.php"
    public_api_user_agent: str = "DeepResearchAgent/0.1"

    arxiv_timeout_seconds: int = Field(default=20, ge=1)
    wikipedia_timeout_seconds: int = Field(default=20, ge=1)

    arxiv_full_text_enabled: bool = True
    arxiv_max_pdf_pages: int = Field(default=20, ge=1)

    search_max_concurrency: int = Field(default=3, ge=1)
    # Bounded parallel page extraction inside one research round.
    extraction_max_concurrency: int = Field(default=5, ge=1)
    web_results_per_query: int = Field(default=5, ge=1)
    arxiv_results_per_query: int = Field(default=3, ge=1)
    wikipedia_results_per_query: int = Field(default=3, ge=1)


class RetrievalSettings(EnvironmentSettings):
    """Configuration for chunking, hybrid retrieval, and LLM-facing context."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="RETRIEVAL_",
    )

    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        min_length=1,
    )

    embedding_device: str = Field(
        default="cpu",
        min_length=1,
    )

    embedding_batch_size: int = Field(
        default=32,
        ge=1,
    )

    # Chunks stay below the embedding model input limit.
    chunk_size: int = Field(
        default=400,
        ge=1,
    )

    # Each retriever can consider more candidates than are ultimately
    # passed to the context builder.
    candidate_k: int = Field(
        default=30,
        ge=1,
    )

    top_k: int = Field(
        default=12,
        ge=1,
    )

    semantic_weight: float = Field(
        default=0.5,
        ge=0.0,
    )

    lexical_weight: float = Field(
        default=0.5,
        ge=0.0,
    )

    rrf_c: int = Field(
        default=60,
        ge=1,
    )

    # The tokenizer is loaded locally only for accurate LLM context accounting.
    context_tokenizer_model: str = Field(
        default="openai/gpt-oss-120b",
        min_length=1,
    )

    # A larger pool lets the LLM evidence scorer do the filtering.
    max_context_tokens: int = Field(
        default=24_000,
        ge=1,
    )

    # Generous so top_k, not the per-source cap, decides what reaches the scorer.
    max_chunks_per_document: int = Field(
        default=8,
        ge=1,
    )


class ResearchLimitSettings(EnvironmentSettings):
    """Safety and resource limits for a single research job."""

    max_question_chars: int = Field(
        default=10_000,
        ge=1,
    )

    max_subquestions: int = Field(
        default=5,
        ge=1,
    )

    max_queries_per_subquestion: int = Field(
        default=3,
        ge=1,
    )

    max_search_query_chars: int = Field(
        default=500,
        ge=1,
    )

    # Short natural queries work better for search engines and retrieval.
    max_search_query_words: int = Field(
        default=12,
        ge=1,
    )

    max_total_queries_per_job: int = Field(
        default=30,
        ge=1,
    )

    max_search_results_per_query: int = Field(
        default=5,
        ge=1,
    )

    # Allow broader first-round collection without increasing the job-wide cap.
    max_extracted_pages_per_round: int = Field(default=30, ge=1)

    # Pre-extraction relevance gate: cosine similarity between a candidate's
    # title + snippet and the subquestion. Calibrate from the gate logs.
    search_relevance_threshold: float = Field(default=0.50, ge=0.0, le=1.0)
    # Keep the best candidates of a subquestion even when all fall below the gate.
    search_relevance_min_keep: int = Field(default=2, ge=0)

    # Reserved for extraction providers where our application controls
    # the raw HTTP download. Tavily performs the remote fetch in Phase 1.
    max_source_download_bytes: int = Field(
        default=5_000_000,
        ge=1,
    )

    min_source_content_chars: int = Field(
        default=80,
        ge=1,
    )

    max_source_content_chars: int = Field(
        default=500_000,
        ge=1,
    )

    min_evidence_relevance: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Minimum LLM relevance score for retaining an original retrieved chunk as evidence.",
    )

    # Maximum number of useful evidence items retained by one research job.
    max_evidence_items: int = Field(
        default=100,
        ge=1,
    )

    max_iterations: int = Field(
        default=3,
        ge=1,
    )

    # LLM intent classifier in the input guardrail (deterministic checks always run).
    input_classifier_enabled: bool = True


class DocumentSettings(EnvironmentSettings):
    """Configuration for private document processing."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="DOCUMENT_",
        extra="ignore",
    )

    max_total_size_bytes: int = Field(
        default=40 * 1024 * 1024,
        gt=0,
    )

    upload_directory: Path = Path("uploaded")

    supported_extensions: tuple[str, ...] = (
        ".pdf",
        ".docx",
        ".txt",
        ".md",
    )

    @property
    def max_total_size_mb(self) -> float:
        """Maximum retained document size in MiB."""
        return self.max_total_size_bytes / (1024 * 1024)


class UploadLimitSettings(EnvironmentSettings):
    """Future Phase 3 limits for user-supplied files."""

    max_files_per_request: int = Field(
        default=10,
        ge=1,
    )

    max_file_size_mb: int = Field(
        default=20,
        ge=1,
    )

    max_total_upload_size_mb: int = Field(
        default=50,
        ge=1,
    )


class VoiceLimitSettings(EnvironmentSettings):
    """Future Phase 5 limits for voice input."""

    max_audio_duration_seconds: int = Field(
        default=300,
        ge=1,
    )

    max_audio_size_mb: int = Field(
        default=25,
        ge=1,
    )


class ObservabilitySettings(EnvironmentSettings):
    """Configuration for application logging and LangSmith tracing."""

    log_level: str = "INFO"
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_project: str = "deep-research-agent"


class Settings(BaseModel):
    """Groups all application settings into one typed configuration object."""

    app: ApplicationSettings
    llm: LLMSettings
    search: SearchSettings
    retrieval: RetrievalSettings
    research_limits: ResearchLimitSettings
    documents: DocumentSettings
    upload_limits: UploadLimitSettings
    voice_limits: VoiceLimitSettings
    observability: ObservabilitySettings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Loads and caches the complete application configuration."""

    return Settings(
        app=ApplicationSettings(),
        llm=LLMSettings(),
        search=SearchSettings(),
        retrieval=RetrievalSettings(),
        research_limits=ResearchLimitSettings(),
        documents=DocumentSettings(),
        upload_limits=UploadLimitSettings(),
        voice_limits=VoiceLimitSettings(),
        observability=ObservabilitySettings(),
    )

