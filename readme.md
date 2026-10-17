# Deep Research Agent

**Deep Research Agent** is an evidence-grounded research application that turns a question into a structured, source-backed report. It plans the investigation, searches public sources and private documents, retrieves only the most relevant passages, evaluates evidence coverage, performs bounded follow-up research, and verifies citations before rendering the final report.

The project addresses a common failure mode in research assistants: sending large amounts of unfiltered content directly to an LLM and trusting a single response. This application separates discovery, extraction, retrieval, evidence validation, coverage analysis, writing, and citation checking into explicit stages that can be inspected, tested, retried, and observed in real time.

> **Project status:** Web, document, and hybrid research flows are implemented. Public-source search, provider fallbacks, shared hybrid retrieval, SQLite-backed jobs, replayable Server-Sent Events (SSE), and the browser UI have offline test coverage. Authentication, durable distributed execution, restart recovery, and production hardening are not yet implemented.

## Table of Contents

- [Problem and Approach](#problem-and-approach)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Research Pipeline](#research-pipeline)
- [Public-Source Pipeline](#public-source-pipeline)
- [Private-Document Pipeline](#private-document-pipeline)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Evidence, Coverage, and Writing](#evidence-coverage-and-writing)
- [LLM and Provider Fallbacks](#llm-and-provider-fallbacks)
- [Jobs, Events, and UI Updates](#jobs-events-and-ui-updates)
- [State and Resource Lifetimes](#state-and-resource-lifetimes)
- [Getting Started](#getting-started)
- [Usage and API Examples](#usage-and-api-examples)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Current Limitations](#current-limitations)
- [Contributing](#contributing)
- [License](#license)

## Problem and Approach

| Research problem | How the project addresses it |
|---|---|
| A broad question is difficult to answer in one prompt. | The planner converts the question into bounded subquestions and objectives. |
| Search results contain duplicates, snippets, boilerplate, and irrelevant text. | Provider adapters normalize results, canonical identifiers merge duplicates, deterministic cleaning removes noise, and retrieval selects relevant passages. |
| One provider can be unavailable or rate-limited. | Search, extraction, and LLM calls use ordered fallbacks with bounded retries and circuit breakers. |
| Public and private sources require different handling. | Web content and private documents use separate retrieval sessions while sharing one chunker and one hybrid-ranking implementation. |
| An LLM may stop too early or repeat broad searches. | A coverage evaluator identifies specific gaps, and a deterministic router permits only bounded, gap-targeted follow-up rounds. |
| Generated claims may not be supported by cited evidence. | The writer can reference only accepted evidence, and the citation checker removes unsupported findings before final rendering. |
| Long-running work is hard to understand from a request spinner. | Persisted stage events drive a live, replayable SSE timeline in the UI. |

## Key Features

- **Multi-stage LangGraph workflow:** Explicit nodes handle input validation, planning, query generation, source selection, retrieval, evidence extraction, coverage evaluation, writing, citation checking, and rendering.
- **Three research modes:** Run web-only, document-only, or hybrid research from the same API and UI.
- **Parallel public search:** Search the general web, arXiv, and Wikipedia concurrently for every generated query.
- **Canonical deduplication:** Merge equivalent pages, papers, and articles using canonical IDs, URLs, arXiv IDs, DOIs, and Wikipedia page IDs.
- **Search fallback:** Retry Tavily and fall back to Brave when the configured Brave credentials are available.
- **Extraction fallback:** Use provider-native arXiv and Wikipedia extractors; retry Tavily extraction and fall back to Firecrawl for general webpages when configured.
- **Deterministic source cleaning:** Normalize Unicode, whitespace, HTML, duplicated lines, PDF hyphenation, and content length without spending an LLM call.
- **Shared retrieval pipeline:** Chonkie creates structure-aware chunks, a dense in-memory index and BM25 search the same corpus, and weighted Reciprocal Rank Fusion combines their rankings.
- **Bounded LLM context:** Enforce token limits and per-document diversity before sending retrieved text to the evidence extractor.
- **Private-document provenance:** Preserve document, filename, page, section, character-offset, and chunk metadata through retrieval, evidence, and final citations.
- **Evidence coverage loop:** Detect missing, partial, conflicting, and supported objectives, then research only actionable gaps while budgets permit.
- **Citation-safe reporting:** Validate evidence references deterministically, perform semantic support checks, remove unsupported findings, and render Markdown from the validated report.
- **Ordered LLM fallback:** Configure OpenAI, Gemini, and Groq independently for each logical role, with private-data provider allowlisting.
- **Persistent job history:** Store asynchronous job status, final results, errors, and ordered progress events in SQLite.
- **SSE-first UI:** Stream replayable progress events, recover the latest job after refresh, and use status polling only when SSE is unavailable.
- **Structured observability:** Emit JSON logs with job, session, node, iteration, provider, role, status, and sanitized failure metadata.

## Tech Stack

| Area | Technology |
|---|---|
| Language | Python 3.13+, JavaScript, HTML, CSS |
| API | FastAPI, Uvicorn, Pydantic, pydantic-settings |
| Workflow | LangGraph |
| LLM abstraction | LangChain |
| LLM providers | OpenAI, Google Gemini, Groq |
| Public search | Tavily, Brave Search, arXiv API, MediaWiki API |
| Content extraction | Tavily Extract, Firecrawl, pypdf, MediaWiki extracts |
| Document parsing | pypdf, python-docx, built-in text parsing |
| Chunking | Chonkie RecursiveChunker |
| Semantic retrieval | Hugging Face embeddings, LangChain InMemoryVectorStore |
| Lexical retrieval | BM25 via rank-bm25 |
| Rank fusion | LangChain EnsembleRetriever with weighted Reciprocal Rank Fusion |
| Persistence | SQLite with aiosqlite and WAL mode |
| Live updates | Server-Sent Events |
| Frontend | Framework-free HTML, CSS, and JavaScript |
| Testing | pytest with fake providers and offline dependencies |
| Observability | Structured JSON logging with job, session, node, iteration, role, and provider metadata |

## Architecture

The architecture below reflects the registered graph, state schema, service composition, and runtime resource lifetimes in the current source.

### System Overview

```mermaid
flowchart LR
    Entry[src.server:app = create_app]
    User[Browser or API client]
    API[FastAPI routes]
    Service[ResearchService preflight and job lifecycle]
    Invoke{Execution mode}
    Graph[14-node compiled LangGraph]
    Providers[Search, extraction, and LLM providers]
    Retrieval[Shared retrieval implementation]
    JobResources[Job-owned private retrieval session]
    Store[(SQLite jobs and events)]
    Uploads[(Local uploaded files)]
    UI[SSE timeline with status fallback]

    Entry --> API
    User -->|Create job or upload file| API
    API --> Service
    Service -->|Validate selected documents before LLM work| Invoke
    Invoke -->|Synchronous graph.ainvoke| Graph
    Invoke -->|Persisted graph.astream_events| Graph
    Graph --> Providers
    Graph --> Retrieval
    Graph --> JobResources
    Graph --> Uploads
    Service --> Store
    Graph -->|Sanitized lifecycle events| Store
    Store -->|Status, replay, and final result| API
    API --> UI --> User
```

### Composition and Dependency Flow

`src/server.py` is the application composition root. It creates expensive or shared dependencies once and injects them into the graph:

1. Load typed settings.
2. Configure structured logging.
3. Create the composite public search provider.
4. Create the provider-aware source extractor and deterministic cleaner.
5. Create the SQLite job store.
6. Create one shared embedding service and local tokenizer.
7. Create the retrieval-session factory.
8. Create the application-scoped document service.
9. Create the role-based LLM service.
10. Build and compile the LangGraph workflow.
11. Create `ResearchService`, register routes, and mount the frontend.

This keeps provider construction out of graph nodes. Nodes depend on small application-owned interfaces, which makes them easier to test with fakes.

Only the upload-facing `DocumentService` is stored in `app.state`. Retrieval sessions are created per job or per public retrieval round and reach graph nodes through injected dependencies and `RunnableConfig`, not global application state.

### Code Request Flow

```mermaid
sequenceDiagram
    participant B as Browser
    participant R as FastAPI route
    participant S as ResearchService
    participant DB as SQLite
    participant G as LangGraph
    participant P as PipelineRecorder

    B->>R: POST /api/research/jobs
    R->>S: start_job(question, source_selection)
    S->>S: Validate selected document availability
    S->>DB: Insert queued job
    S-->>B: 202 + job_id
    S->>G: run_graph_with_events(state, job_resources)
    G->>P: Node lifecycle events
    P->>DB: Persist sanitized event
    B->>R: GET /events (SSE)
    R->>DB: Replay events after cursor
    DB-->>B: stage_started / completed / failed
    G-->>S: Final WorkflowState
    S->>DB: Atomically store result + terminal event
    S->>S: Clear job resources and LLM provider state
    B->>R: GET /status/{job_id}
    R->>DB: Read terminal job
    DB-->>B: Final structured report
```

## Research Pipeline

### LangGraph Node Interconnection

```mermaid
flowchart TD
    Start([START]) --> Guardrail[input_guardrail]
    Guardrail -->|valid| Planner[research_planner]
    Guardrail -->|terminal error| End([END])
    Planner -->|plan created| Query[research_query_generator]
    Planner -->|terminal error| End
    Query --> Select[research_source_selection]

    Select -->|web or hybrid| Search[research_web_search]
    Select -->|documents only| Retrieve[research_retrieval]
    Select -->|terminal error| End

    Search --> Process[research_source_processor]
    Process --> Retrieve
    Retrieve --> Evidence[research_evidence_extractor]
    Evidence --> Coverage[research_coverage_evaluator]
    Coverage --> Router[research_routing]

    Router -->|actionable gaps and budget remains| Query
    Router -->|coverage sufficient or stop limit| Prepare[writer_prepare_evidence]
    Router -->|terminal error| End

    Prepare --> Synthesis[writer_synthesis]
    Synthesis -->|draft created| Citations[writer_citation_checker]
    Synthesis -->|terminal error| End
    Citations -->|validated report| Finalize[finalize_report]
    Citations -->|terminal error| End
    Finalize --> End
```

Hybrid mode follows the public-search branch, but `research_source_selection` also prepares a private-document retrieval session. The retrieval node queries both sessions and preserves their results separately until the evidence boundary.

### Node Responsibilities

| Node | Problem solved | Implementation |
|---|---|---|
| `input_guardrail` | Invalid questions or incomplete source selections should fail before provider calls. | Validates question length, allowed sources, owner identity, and document selection. |
| `research_planner` | A broad question needs a research plan. | Uses structured LLM output to create bounded subquestions and objectives. |
| `research_query_generator` | Each objective needs searchable wording. | Creates initial queries or gap-focused follow-up queries and applies job query limits. |
| `research_source_selection` | Source modes need different runtime resources. | Authorizes selected documents and builds one job-scoped document retrieval session. `route_after_source_selection` chooses the public-search or document-only edge. |
| `research_web_search` | One public search source is incomplete and may fail. | Runs current-round queries through the parallel composite provider and retains normalized results and diagnostics. |
| `research_source_processor` | Search results are not evidence-ready documents. | Loads provider-native or webpage content, cleans it, deduplicates it, and creates normalized `Source` objects. |
| `research_retrieval` | Entire sources are too large and noisy for an LLM. | Builds a current-round public retrieval session, reuses the job-owned private session, and returns separate bounded web/document contexts. |
| `research_evidence_extractor` | Retrieved text must become explicit claims with provenance. | Produces schema-validated evidence linked to chunks, sources, and research objectives; normalizes duplicates. |
| `research_coverage_evaluator` | The system must know what remains unanswered. | Rates each subquestion, validates referenced evidence IDs, and emits actionable coverage gaps. |
| `research_routing` | An LLM should not control an unbounded loop. | Deterministically checks coverage, iteration count, query usage, extraction attempts, elapsed time, and evidence gain. |
| `writer_prepare_evidence` | The writer needs a relevant, balanced evidence set. | Ranks evidence by relevance, gives each subquestion an opportunity to contribute, and stores only selected evidence IDs in writer state. |
| `writer_synthesis` | Evidence must become a readable report without inventing claims. | Produces a structured draft from selected evidence and records unresolved gaps as limitations. |
| `writer_citation_checker` | Draft citations may be missing or unsupported. | Performs deterministic reference checks and semantic evidence-to-claim validation; unsupported findings are removed. |
| `finalize_report` | Output formatting should not change validated meaning. | Deterministically renders the validated `ResearchReport` into Markdown. |

### Bounded Research Loop

The research loop continues only when coverage contains actionable gaps and all configured budgets still allow another round.

```mermaid
stateDiagram-v2
    [*] --> ResearchRound
    ResearchRound --> CoverageEvaluation
    CoverageEvaluation --> WriteReport: coverage sufficient
    CoverageEvaluation --> WriteReport: no actionable gaps
    CoverageEvaluation --> WriteReport: iteration/query/extraction/time budget reached
    CoverageEvaluation --> WriteReport: no evidence improvement
    CoverageEvaluation --> GapTargetedQueries: unresolved gaps + budget available
    GapTargetedQueries --> ResearchRound
    WriteReport --> [*]
```

Supported stop reasons include:

- **`coverage_sufficient`:** Evidence adequately supports the planned objectives.
- **`max_iterations`:** The bounded round limit was reached.
- **`query_budget_exhausted`:** The job used its permitted query count.
- **`extraction_budget_exhausted`:** The job reached its page-processing limit.
- **`time_budget_exhausted`:** Research exceeded its elapsed-time budget.
- **`no_evidence_improvement`:** Additional rounds stopped contributing newly referenced evidence.
- **`no_actionable_gaps`:** Coverage is incomplete, but no useful follow-up gap remains.

## Public-Source Pipeline

### Search, Deduplication, Extraction, and Cleaning

```mermaid
flowchart LR
    Query[Generated SearchQuery]

    subgraph Parallel Search
        General[General web branch]
        Arxiv[arXiv API]
        Wiki[Wikipedia API]
    end

    Query --> General
    Query --> Arxiv
    Query --> Wiki
    General --> Tavily[Tavily search]
    Tavily -->|eligible failure| Brave[Brave fallback]

    Tavily --> Merge[Canonical deduplication]
    Brave --> Merge
    Arxiv --> Merge
    Wiki --> Merge

    Merge --> Route{Provider-aware extraction}
    Route -->|Tavily or Brave result| TavilyExtract[Tavily Extract]
    TavilyExtract -->|eligible failure| Firecrawl[Firecrawl fallback]
    Route -->|arXiv result| ArxivLoad[Bounded PDF or abstract]
    Route -->|Wikipedia result| WikiLoad[Canonical MediaWiki article]

    TavilyExtract --> Cleaner[Deterministic SourceCleaner]
    Firecrawl --> Cleaner
    ArxivLoad --> Cleaner
    WikiLoad --> Cleaner
    Cleaner --> Source[Normalized Source]
```

#### Search Behavior

- **Parallel branches:** General web, arXiv, and Wikipedia searches run concurrently under a shared semaphore.
- **Per-provider limits:** Each branch keeps its own result cap instead of comparing incompatible provider scores.
- **Partial success:** A failed branch does not discard useful results from healthy branches. The query fails only when every branch fails.
- **General-search fallback:** Tavily retries first. Brave activates only for failures classified as eligible for fallback and only when `BRAVE_API_KEY` is configured.
- **Canonical merging:** Duplicate identities are detected from canonical IDs, normalized URLs, DOI values, arXiv IDs without version suffixes, Wikipedia page IDs, and Wikipedia titles.
- **Discovery provenance:** When multiple providers discover the same source, the representative result retains all discovery metadata.

#### Extraction Behavior

- **General webpages:** Tavily Extract is primary; Firecrawl is the optional fallback.
- **arXiv:** The native adapter downloads a bounded PDF and reuses the existing PDF parser. If full text is disabled or unusable, it falls back to the abstract.
- **Wikipedia:** The native adapter requests canonical plain text, redirects, page ID, and revision metadata from MediaWiki.
- **Circuit breaking:** Repeated eligible failures temporarily open a provider circuit so every URL or query does not repeatedly call an unhealthy service.

#### Deterministic Cleaning

`SourceCleaner` runs after extraction and before retrieval. It:

- normalizes Unicode and line endings;
- removes control characters;
- converts useful HTML structure into readable text;
- drops script, style, navigation, footer, and other boilerplate elements;
- joins PDF-style line-break hyphenation;
- collapses whitespace and duplicate adjacent lines;
- enforces minimum and maximum content lengths; and
- records cleaner version and size metadata for traceability.

The cleaner does not call an LLM.

## Private-Document Pipeline

### Upload and Research-Time Processing

```mermaid
flowchart TD
    Pick[User selects PDF, DOCX, TXT, or Markdown]
    Upload[POST /documents]
    Store[Stream file to local storage]
    Registry[Store owner-scoped metadata in memory]
    Job[Start document or hybrid research job]
    Authorize[Authorize owner + document IDs]
    Parse[Parse into provenance-aware blocks]
    Adapt[Convert blocks to RetrievalDocument]
    Shared[Shared RetrievalSession]
    Chunk[RetrievalChunker]
    Index[Dense + BM25 indexes]
    Search[Retrieve for every subquestion]
    Cleanup[Clear job-scoped chunks and indexes]

    Pick --> Upload --> Store --> Registry
    Registry --> Job --> Authorize --> Parse --> Adapt --> Shared
    Shared --> Chunk --> Index --> Search --> Cleanup
```

Uploads and indexing are intentionally separate:

1. **Upload time:** The service validates the extension, streams the file to disk, computes a SHA-256 hash, and stores metadata. It does not parse, chunk, embed, or index the file.
2. **Job start:** `research_source_selection` verifies that every selected document belongs to the request owner.
3. **Parsing:** PDF pages, DOCX paragraphs, or text content become normalized blocks with available page and character provenance.
4. **Shared retrieval preparation:** Blocks become `RetrievalDocument` objects and enter the same `RetrievalSession` implementation used for public content, but in a separate job-owned session.
5. **Reuse during the job:** One private-document session serves every subquestion and later coverage iteration.
6. **Cleanup:** Job-scoped chunks and indexes are cleared when the workflow completes, fails, or is cancelled.

The original uploaded file remains on local disk unless the optional deletion call in `DocumentService` is enabled. Document metadata is held by an application-scoped in-memory registry, so it does not survive a server restart.

### Hybrid Source Isolation

```mermaid
flowchart LR
    WebSources[Cleaned public Sources] --> WebSession[Public RetrievalSession]
    PrivateFiles[Selected private documents] --> DocumentSession[Private RetrievalSession]
    Questions[Subquestions and queries] --> WebSession
    Questions --> DocumentSession
    WebSession --> WebResults[web_retrieval_results]
    DocumentSession --> DocumentResults[document_retrieval_results]
    WebResults --> Evidence[Evidence extractor]
    DocumentResults --> Evidence
    Evidence --> Accepted[Normalized evidence with provenance]
```

Public and private chunks are not inserted into one long-lived shared vector database. Separate sessions reduce accidental cross-job or cross-source leakage, while the evidence extractor provides the single controlled merge point.

## Retrieval Pipeline

```mermaid
flowchart TD
    Docs[RetrievalDocument corpus]
    Chunker[Chonkie RecursiveChunker]
    Chunks[Provenance-aware RetrievalChunks]
    Dense[Semantic vector retrieval]
    BM25[Lexical BM25 retrieval]
    RRF[Weighted Reciprocal Rank Fusion]
    Diversity[Per-document diversity filter]
    Budget[Token-bounded context builder]
    Context[Evidence-extraction context]

    Docs --> Chunker --> Chunks
    Chunks --> Dense
    Chunks --> BM25
    Dense --> RRF
    BM25 --> RRF
    RRF --> Diversity --> Budget --> Context
```

### Why Hybrid Retrieval

- **Semantic retrieval** finds conceptually related passages even when wording differs from the query.
- **BM25 retrieval** preserves exact terms, names, identifiers, and domain-specific keywords.
- **Reciprocal Rank Fusion** combines both ranked lists without requiring their raw scores to be comparable.
- **Diversity limits** prevent one long source from occupying the entire context.
- **Token accounting** stops context growth before the evidence LLM call.

`src/retrieval/chunking/retrieval_chunker.py` is the only chunking implementation. Public sources and private documents both pass through it, ensuring consistent chunk boundaries and provenance behavior.

### Data Transformation

```text
SearchResult
    -> Source
    -> RetrievalDocument
    -> RetrievalChunk
    -> RetrievedChunk
    -> RetrievalResult
    -> Evidence
    -> ReportFinding
    -> ResearchReport
```

Each transformation narrows the data while retaining identifiers needed to trace a final finding back to its source and chunk.

## Evidence, Coverage, and Writing

### Evidence Boundary

The evidence extractor receives only retrieved contexts—not complete source corpora. It returns schema-validated evidence containing:

- the supported claim or relevant content;
- the associated subquestion;
- the source and chunk identifiers;
- relevance and provenance metadata; and
- private-document locations when applicable.

Evidence from web and document results is normalized into one evidence collection. Exact normalized duplicates within the same subquestion are not added twice; every retained item keeps the provenance of its supporting chunk and source.

### Coverage Evaluation

The coverage evaluator checks every planned subquestion and classifies its support. It can identify:

- supported objectives;
- partially supported objectives;
- missing evidence;
- conflicting evidence; and
- actionable follow-up gaps.

The deterministic router—not the LLM—decides whether those gaps justify another iteration.

### Writer and Citation Pipeline

```mermaid
flowchart LR
    Evidence[Accepted evidence]
    Select[Relevance-ranked evidence selection]
    Draft[Structured draft report]
    References[Deterministic ID checks]
    Semantic[Semantic citation verification]
    Filter[Remove unsupported findings]
    Render[Deterministic Markdown rendering]

    Evidence --> Select --> Draft --> References --> Semantic --> Filter --> Render
```

The writer cannot cite evidence it did not receive. The citation checker verifies that referenced evidence and sources exist, then checks whether the cited evidence actually supports each finding. Findings that fail validation do not reach the final report.

## LLM and Provider Fallbacks

### Role-Based LLM Selection

Nodes request logical roles rather than concrete provider SDKs:

- `planner`
- `query_generator`
- `evidence_extractor`
- `coverage_evaluator`
- `writer`
- `citation_checker`

Each role can define its own ordered provider/model chain.

These are structured LLM calls, not autonomous agent executors. The graph contains no ReAct loop, LLM tool-selection loop, nested LangGraph, human interrupt, or checkpoint-backed resume path. Search, extraction, parsing, retrieval, persistence, and rendering are invoked directly by application code.

```mermaid
flowchart LR
    Role[Logical LLM role]
    Policy{Private data?}
    OpenAI[OpenAI model]
    Gemini[Gemini model]
    Groq[Groq model]
    Result[Schema-validated output]

    Role --> Policy
    Policy -->|Filter by allowlist| OpenAI
    OpenAI -->|eligible failure| Gemini
    Gemini -->|eligible failure| Groq
    OpenAI --> Result
    Gemini --> Result
    Groq --> Result
```

The service:

- preserves configured provider order;
- filters providers through `LLM_PRIVATE_DATA_PROVIDERS` for document and hybrid jobs;
- retries malformed structured output a bounded number of times;
- fails over on classified availability, authentication, rate-limit, timeout, server, and provider-specific structured-output failures;
- keeps invalid prompts, policy failures, oversized requests, and context-length failures terminal when another provider is unlikely to help;
- remembers the successful provider independently for later calls to the same role in the same job; and
- releases job-specific provider state at job completion.

### Search and Extraction Resilience

| Capability | Primary | Fallback | Activation |
|---|---|---|---|
| General web search | Tavily | Brave | Eligible primary failure and configured Brave key |
| General webpage extraction | Tavily Extract | Firecrawl | Eligible primary failure or unusable extraction and configured Firecrawl key |
| arXiv content | Bounded PDF | arXiv abstract | PDF disabled, unavailable, invalid, oversized, or unparsable |
| Wikipedia content | MediaWiki API | None | Branch remains recoverable when other public branches succeed |
| Structured LLM role | First configured eligible provider | Next configured provider | Classified call failure and private-data policy eligibility |

## Jobs, Events, and UI Updates

### Persistence and Streaming

SQLite stores:

- **`research_jobs`:** IDs, question, allowed sources, lifecycle status, final result, error message, and timestamps.
- **`research_events`:** Ordered, per-job progress events keyed by `(job_id, sequence)`.

The terminal job update and terminal event are committed in one transaction so the UI cannot observe a terminal event without a matching result.

Representative event types:

| Event | Meaning |
|---|---|
| `workflow_started` | Graph execution began. |
| `stage_started` | A named pipeline node started. |
| `stage_completed` | A node completed successfully. |
| `stage_failed` | A node returned or raised a terminal failure. |
| `research_retry` | The router scheduled a gap-targeted round. |
| `research_stopped` | Research ended and writing can begin. |
| `workflow_completed` | A final report was persisted. |
| `workflow_failed` | A terminal error was persisted. |

### SSE-First Browser Behavior

```mermaid
flowchart TD
    Open[Open or restore job]
    Status[Fetch status once]
    SSE[Connect EventSource]
    Replay[Replay persisted events after cursor]
    Live[Receive live events]
    Terminal{Terminal event?}
    Final[Fetch final status and report]
    Failed{SSE unavailable?}
    Poll[Backoff status polling]

    Open --> Status --> SSE --> Replay --> Live --> Terminal
    Terminal -->|yes| Final
    Terminal -->|no| Live
    SSE --> Failed
    Failed -->|yes| Poll
    Poll -->|SSE reconnects| Live
    Poll -->|job terminal| Final
```

Progress events expose counts, stage names, provider names, fallback activation, coverage status, and sanitized error summaries. They do not persist prompts, source bodies, evidence text, private document IDs, credentials, or final report contents.

## State and Resource Lifetimes

Understanding resource scope is important because the project intentionally avoids one global mixed-source vector database.

| Scope | Resources | Lifetime |
|---|---|---|
| Application process | Settings, provider clients, embedding service, tokenizer, retrieval factory, document service, in-memory document registry | Server process |
| Research job | LangGraph state, private-document retrieval session, LLM provider preference/failure state | One job |
| Public retrieval invocation | Current-round public chunks, dense index, BM25 index, retrieval contexts | Built for one retrieval-node invocation; the current explicit `clear()` call is disabled |
| Persistent local storage | SQLite jobs/events, uploaded source files | Until manually removed |
| Browser | Temporary owner ID and last job ID | Browser local storage |

### Workflow State Ownership

| State section | Owns |
|---|---|
| Root | Job/session IDs, question, owner ID, selected source mode, selected document IDs, structured errors |
| `research` | Plan, queries, search results, normalized sources, separate web/document retrieval results, evidence, coverage, iteration count, stop reason |
| `research_control` | Budgets, usage, pending gaps, routing decision, and round history |
| `writer` | Selected evidence IDs, draft report, citation result, and final report |

Non-serializable retrieval indexes stay in `ResearchJobResources`, outside graph state.

`WorkflowState` is a `TypedDict` without `Annotated` reducers. A returned top-level key replaces that key; nodes preserve nested `research` and `writer` fields with explicit dictionary spreads, append or deduplicate lists themselves, and update `ResearchControlState` through Pydantic `model_copy()`. The private retrieval session and per-job LLM provider health remain runtime resources rather than serializable graph state.

## Getting Started

### Prerequisites

- **Python:** 3.13 or newer, matching `pyproject.toml`.
- **Package tooling:** `pip` and `venv`.
- **Tavily:** A Tavily API key is currently required by application settings for general search and extraction.
- **LLM access:** At least one supported provider key and model ID: OpenAI, Google Gemini, or Groq.
- **Network access:** Required for public research and the first download of the local embedding model and tokenizer.

Brave and Firecrawl credentials are optional. arXiv and Wikipedia use public APIs and do not require API keys.

### Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/Ishwaridutt/deep-research-agent.git
cd deep-research-agent
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### Environment Configuration

Copy the provided template:

```bash
cp .env.example .env
```

Configure at least Tavily and one LLM provider. This single-provider example uses Groq:

```dotenv
TAVILY_API_KEY=replace-with-your-tavily-key
GROQ_API_KEY=replace-with-your-groq-key

LLM_PROVIDER=groq
LLM_PROVIDER_ORDER=groq
LLM_PRIVATE_DATA_PROVIDERS=groq

PLANNER_MODEL=groq:replace-with-supported-model-id
QUERY_GENERATOR_MODEL=groq:replace-with-supported-model-id
EVIDENCE_MODEL=groq:replace-with-supported-model-id
COVERAGE_MODEL=groq:replace-with-supported-model-id
WRITER_MODEL=groq:replace-with-supported-model-id
CITATION_MODEL=groq:replace-with-supported-model-id

LANGSMITH_TRACING=false
```

To enable the full LLM fallback chain, configure all provider keys and give each role a comma-separated chain:

```dotenv
OPENAI_API_KEY=replace-with-your-openai-key
GOOGLE_API_KEY=replace-with-your-google-key
GROQ_API_KEY=replace-with-your-groq-key

LLM_PROVIDER_ORDER=openai,google_genai,groq
LLM_PRIVATE_DATA_PROVIDERS=openai,google_genai,groq

PLANNER_MODEL=openai:OPENAI_MODEL_ID,google_genai:GEMINI_MODEL_ID,groq:GROQ_MODEL_ID
```

Apply the same chain format to the other five role settings. Never commit `.env` or real credentials.

### Running Locally

```bash
source .venv/bin/activate
python -m uvicorn src.server:app --reload
```

Open:

- **Application:** http://127.0.0.1:8000/
- **API documentation:** http://127.0.0.1:8000/docs
- **Health check:** http://127.0.0.1:8000/health

Use one Uvicorn worker. Background tasks, document metadata, and job-owned resources are currently process-local and are not coordinated across workers.

## Usage and API Examples

### Browser Workflow

1. Enter a research question.
2. Select **Web**, **Documents**, or **Hybrid**.
3. For document or hybrid research, choose one or more `.pdf`, `.docx`, `.txt`, or `.md` files, or select existing uploads from the current process.
4. Select **Start research**. Pending files upload automatically before the research job starts.
5. Follow the live pipeline, provider progress, coverage rounds, and final report.

### API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Serve the browser workspace. |
| `GET` | `/health` | Return application health. |
| `POST` | `/documents` | Upload one owner-scoped private document. |
| `GET` | `/documents` | List documents for an owner. |
| `POST` | `/api/research` | Run a synchronous research workflow. |
| `POST` | `/api/research/jobs` | Queue an asynchronous persisted job. |
| `GET` | `/api/research/jobs/status/{job_id}` | Read lifecycle state and terminal result. |
| `GET` | `/api/research/jobs/{job_id}/events` | Replay and stream SSE events. |

### Start a Web Research Job

```bash
curl -X POST http://127.0.0.1:8000/api/research/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "How are agentic AI systems changing software engineering?",
    "source_selection": {
      "mode": "web",
      "document_ids": []
    }
  }'
```

Example response:

```json
{
  "job_id": "JOB-example",
  "research_session_id": "RS-example",
  "status": "queued"
}
```

### Upload and Use a Private Document

Use the same owner ID for upload, listing, and research:

```bash
OWNER_ID="local-example-owner"

curl -X POST http://127.0.0.1:8000/documents \
  -H "X-Owner-Id: ${OWNER_ID}" \
  -F "file=@./example.pdf"
```

Copy the returned `document_id`, then create a hybrid job:

```bash
DOCUMENT_ID="returned-document-id"

curl -X POST http://127.0.0.1:8000/api/research/jobs \
  -H 'Content-Type: application/json' \
  -H "X-Owner-Id: ${OWNER_ID}" \
  -d "{
    \"question\": \"Compare this document with current public research.\",
    \"source_selection\": {
      \"mode\": \"hybrid\",
      \"document_ids\": [\"${DOCUMENT_ID}\"]
    }
  }"
```

Use `"mode": "documents"` for private-document-only research.

### Check Job Status

```bash
curl http://127.0.0.1:8000/api/research/jobs/status/JOB-example
```

Status is one of `queued`, `running`, `completed`, or `failed`. A terminal response includes the structured execution result and either the validated report or structured errors.

### Stream Pipeline Events

```bash
curl -N http://127.0.0.1:8000/api/research/jobs/JOB-example/events
```

Replay only events after sequence 20:

```bash
curl -N 'http://127.0.0.1:8000/api/research/jobs/JOB-example/events?after=20'
```

The endpoint also honors the `Last-Event-ID` header used by browser `EventSource` reconnections.

### Run Synchronously

```bash
curl -X POST http://127.0.0.1:8000/api/research \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "What are the major approaches to evaluating AI agents?",
    "source_selection": {"mode": "web", "document_ids": []}
  }'
```

The connection remains open until the workflow completes. Prefer the job API for the interactive UI and long-running research.

## Configuration

Use `.env.example` as the complete configuration reference.

### LLM Settings

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `GROQ_API_KEY` | Provider credentials. |
| `LLM_PROVIDER` | Default provider for unqualified model names. |
| `LLM_PROVIDER_ORDER` | Global provider preference order. |
| `LLM_PRIVATE_DATA_PROVIDERS` | Providers allowed to receive private-document content. |
| `*_MODEL` | Provider-qualified model chain for each logical role. |
| `LLM_TIMEOUT_SECONDS` | Per-provider timeout. |
| `LLM_MAX_RETRIES` | Provider SDK retry count. |
| `LLM_STRUCTURED_OUTPUT_METHOD` | LangChain structured-output method. |
| `LLM_STRUCTURED_OUTPUT_ATTEMPTS` | Bounded parsing/schema attempts. |

### Search and Extraction Settings

| Variable | Purpose |
|---|---|
| `TAVILY_API_KEY` | Required general search and extraction credential. |
| `BRAVE_API_KEY` | Optional general-search fallback credential. |
| `FIRECRAWL_API_KEY` | Optional webpage-extraction fallback credential. |
| `ARXIV_ENABLED`, `WIKIPEDIA_ENABLED` | Enable native public search branches. |
| `ARXIV_FULL_TEXT_ENABLED` | Attempt bounded arXiv PDF extraction before abstract fallback. |
| `SEARCH_MAX_CONCURRENCY` | Maximum concurrent provider branches. |
| `*_RESULTS_PER_QUERY` | Per-provider result caps. |
| `PROVIDER_MAX_RETRIES` | Application-level provider retries. |
| `PROVIDER_CIRCUIT_*` | Circuit-breaker threshold and recovery window. |

### Retrieval Settings

All retrieval variables use the `RETRIEVAL_` prefix.

| Variable | Default | Purpose |
|---|---:|---|
| `RETRIEVAL_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local semantic embedding model. |
| `RETRIEVAL_EMBEDDING_DEVICE` | `cpu` | Embedding execution device. |
| `RETRIEVAL_CHUNK_SIZE` | `400` | Chonkie chunk size. |
| `RETRIEVAL_CANDIDATE_K` | `20` | Candidates requested from each retrieval path. |
| `RETRIEVAL_TOP_K` | `8` | Fused passages considered for context. |
| `RETRIEVAL_SEMANTIC_WEIGHT` | `0.5` | Dense-ranking contribution. |
| `RETRIEVAL_LEXICAL_WEIGHT` | `0.5` | BM25-ranking contribution. |
| `RETRIEVAL_RRF_C` | `60` | Reciprocal Rank Fusion constant. |
| `RETRIEVAL_MAX_CONTEXT_TOKENS` | `3000` | Maximum selected retrieval context. |
| `RETRIEVAL_MAX_CHUNKS_PER_DOCUMENT` | `3` | Per-document diversity cap. |

### Research and Document Limits

| Variable | Purpose |
|---|---|
| `MAX_SUBQUESTIONS` | Maximum planning breadth. |
| `MAX_QUERIES_PER_SUBQUESTION` | Query-generation limit per objective. |
| `MAX_TOTAL_QUERIES_PER_JOB` | Job-wide query budget. |
| `MAX_SOURCE_DOWNLOAD_BYTES` | Bounded source/PDF download size. |
| `MIN_SOURCE_CONTENT_CHARS`, `MAX_SOURCE_CONTENT_CHARS` | Cleaner acceptance limits. |
| `MIN_EVIDENCE_RELEVANCE` | Minimum accepted evidence relevance. |
| `MAX_EVIDENCE_ITEMS` | Maximum accumulated evidence. |
| `MAX_ITERATIONS` | Maximum research rounds. |
| `MAX_WRITER_EVIDENCE_ITEMS`, `MAX_WRITER_EVIDENCE_CHARS` | Writer context bounds. |
| `DOCUMENT_MAX_TOTAL_SIZE_BYTES` | Per-owner upload quota in the in-memory registry. |
| `DOCUMENT_UPLOAD_DIRECTORY` | Local uploaded-file directory. |

## Project Structure

```text
.
├── src/
│   ├── agents/
│   │   ├── research/                 # Planning, search, retrieval, evidence, coverage nodes
│   │   └── writer/                   # Evidence selection, synthesis, citation checking
│   ├── config/
│   │   ├── settings.py               # Typed environment configuration
│   │   └── logging.py                # Structured JSON logging
│   ├── documents/
│   │   ├── document_parser.py        # PDF, DOCX, text, and Markdown parsing
│   │   ├── document_registry.py      # Process-local owner-scoped metadata
│   │   └── local_document_file_store.py
│   ├── graph/
│   │   ├── graph_builder.py          # LangGraph nodes and edges
│   │   ├── graph_routing.py          # Deterministic conditional routing
│   │   └── graph_state.py            # Workflow state ownership
│   ├── guardrails/                   # Input and URL validation
│   ├── llm/
│   │   ├── factory.py                # Provider/model-chain construction
│   │   ├── roles.py                  # Logical LLM responsibilities
│   │   └── service.py                # Structured output, policy, and fallback
│   ├── models/                       # Pydantic domain and API contracts
│   ├── persistence/
│   │   └── research_job_store.py     # SQLite jobs and ordered events
│   ├── prompts/                      # Role-specific prompt builders
│   ├── rendering/                    # Deterministic report rendering
│   ├── retrieval/
│   │   ├── chunking/                 # Single shared RetrievalChunker
│   │   ├── context/                  # Token-bounded context selection
│   │   ├── embeddings/               # Shared embedding service
│   │   └── retrievers/               # BM25, dense retrieval, and RRF
│   ├── routes/                       # Research, document, health, and UI routes
│   ├── services/
│   │   ├── document_service.py       # Upload and research-time document adaptation
│   │   ├── research_service.py       # Job lifecycle and graph execution
│   │   └── research_pipeline_stream_service.py
│   ├── static/                       # Browser workspace
│   ├── tests/                        # Offline focused and integration-style tests
│   ├── tools/
│   │   ├── cleaning/                 # Deterministic SourceCleaner
│   │   ├── extraction/               # Native, web, and fallback extractors
│   │   └── search/                   # Parallel search and provider adapters
│   └── server.py                     # FastAPI composition root
├── docs/                             # Architecture, phase plans, and implementation handoffs
├── data/                             # Runtime SQLite database
├── uploaded/                         # Runtime private uploads; gitignored
├── .env.example                      # Safe configuration template
├── requirements.txt
├── pyproject.toml
└── readme.md
```

### Important Entry Points

| File | Start here when changing |
|---|---|
| `src/server.py` | Application wiring, resource creation, startup, and shutdown |
| `src/graph/graph_builder.py` | Node order or conditional graph edges |
| `src/graph/graph_routing.py` | Research-loop and terminal routing rules |
| `src/graph/graph_state.py` | Data shared between nodes |
| `src/services/research_service.py` | Synchronous/asynchronous job execution |
| `src/services/research_pipeline_stream_service.py` | SSE lifecycle event capture and sanitization |
| `src/retrieval/retrieval_session.py` | Shared indexing and retrieval boundary |
| `src/tools/search/factory.py` | Search-provider composition |
| `src/tools/extraction/factory.py` | Extraction-provider composition |
| `src/static/js/app.js` | UI submissions, SSE connection, recovery, and rendering |

## Testing

Tests must not call real LLM, search, extraction, or embedding providers. The suite uses fake dependencies and disables dotenv loading:

```bash
PYTHON_DOTENV_DISABLED=1 .venv/bin/python -m pytest -q src/tests
```

Additional checks:

```bash
PYTHON_DOTENV_DISABLED=1 .venv/bin/python -m compileall -q src
node --check src/static/js/app.js
git diff --check
```

Focused examples:

```bash
PYTHON_DOTENV_DISABLED=1 .venv/bin/python -m pytest -q src/tests/test_retrieval_pipeline.py
PYTHON_DOTENV_DISABLED=1 .venv/bin/python -m pytest -q src/tests/test_parallel_search.py
PYTHON_DOTENV_DISABLED=1 .venv/bin/python -m pytest -q src/tests/test_provider_fallbacks.py
PYTHON_DOTENV_DISABLED=1 .venv/bin/python -m pytest -q src/tests/test_llm_fallback.py
PYTHON_DOTENV_DISABLED=1 .venv/bin/python -m pytest -q src/tests/test_hybrid_evidence_flow.py
```

The suite changes with the implementation; run the command above for the current result instead of relying on a fixed test count.

## Current Limitations

- **Temporary owner identity:** `X-Owner-Id` is not authentication. Do not expose private-document routes publicly without real identity and authorization.
- **Process-local document metadata:** Uploaded files remain on disk, but the in-memory document registry is lost on restart. Previously uploaded files cannot be selected again without rebuilding metadata.
- **No workflow checkpoint resume:** SQLite preserves jobs and events, not LangGraph execution checkpoints. Queued or running jobs are marked failed after a restart.
- **Single-process execution:** Background tasks, admission control, and runtime resources are not coordinated across Uvicorn workers or machines.
- **In-memory retrieval indexes:** Dense and BM25 indexes are not persisted. The private session is cleared on terminal job paths; explicit public-session cleanup is currently disabled in `research_retrieval_node` and should be restored before production hardening.
- **Writer budget enforcement:** Evidence selection stops after observed item or character totals reach their limits, but `_evidence_fits_budget()` currently returns `True`, so the item that crosses the remaining character budget can still be accepted.
- **No durable task queue:** Jobs run as application-owned asyncio tasks.
- **No global cost or token quota:** Retrieval and writer contexts are bounded, but provider billing, total generated tokens, and account-level quotas are not centrally enforced.
- **Development runtime prints:** Learning-oriented prints currently expose intermediate diagnostics and should be removed or sanitized before production deployment.
- **Local file storage:** Upload lifecycle, retention, encryption, malware scanning, and secure deletion require production policies.
- **No public deployment controls:** Rate limiting, CSRF strategy, access control, secret management, retention, migrations, metrics, and operational alerting remain future work.

## Contributing

Contributions should keep the architecture's existing boundaries intact.

1. Fork the repository and create a focused branch.
2. Inspect existing utilities before adding a parallel implementation.
3. Keep `RetrievalChunker` as the single chunking implementation.
4. Add focused offline tests for changed behavior.
5. Run the full test suite and static checks.
6. Update relevant documentation and submit a pull request describing the problem, design decision, and verification results.

Do not include credentials, `.env`, generated databases, uploaded research files, or provider response payloads in commits or bug reports.

## License

This repository does not currently include a license file. Until a license is added, no open-source reuse rights are granted by default. Add a recognized license such as MIT, Apache-2.0, or GPL-3.0 before distributing the project as open source.

---

**Design principle:** Spend model context on selected evidence and synthesis—not entire webpages or repeated broad searches—and make every decision to continue, stop, retry, or reject a claim observable and bounded.
