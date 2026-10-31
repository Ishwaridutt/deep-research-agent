"use strict";

const $ = id => document.getElementById(id);

const ui = {
    form: $("research-form"),
    question: $("research-question"),
    submit: $("research-button"),
    sourceIndicator: $("source-indicator-text"),
    documentPanel: $("document-panel"),
    documentUpload: $("document-upload"),
    refreshDocuments: $("refresh-documents"),
    documentStatus: $("document-status"),
    documentList: $("document-list"),
    status: $("status-panel"),
    spinner: $("status-spinner"),
    statusIcon: $("status-static-icon"),
    statusTitle: $("status-title"),
    statusMessage: $("status-message"),
    connection: $("connection-state"),
    pipeline: $("pipeline-panel"),
    timeline: $("pipeline-timeline"),
    elapsed: $("workflow-elapsed"),
    pipelineJob: $("workflow-job"),
    copyLink: $("copy-job-link"),
    coverageStatus: $("coverage-status"),
    coverageDescription: $("coverage-description"),
    coverageGapCount: $("coverage-gap-count"),
    coverageAssessmentCount: $("coverage-assessment-count"),
    activity: $("activity-feed"),
    activityEmpty: $("activity-empty"),
    eventCount: $("event-count"),
    result: $("result-panel"),
    reportHeading: $("report-heading"),
    resultStatus: $("result-status"),
    metadata: $("job-metadata"),
    summary: $("report-summary"),
    findings: $("report-findings"),
    limitations: $("report-limitations"),
    unverifiedSection: $("unverified-section"),
    unverified: $("report-unverified"),
    sources: $("report-sources"),
    errorsSection: $("workflow-errors-section"),
    errorsSummary: $("workflow-errors-summary"),
    errors: $("workflow-errors"),
    markdownSection: $("markdown-section"),
    markdown: $("report-markdown"),
};

const STAGES = {
    input_guardrail: "Input validation",
    research_planner: "Research planning",
    research_query_generator: "Generate search queries",
    research_source_selection: "Prepare selected sources",
    research_web_search: "Search public sources",
    research_source_processor: "Extract public source content",
    research_retrieval: "Find relevant passages · Public sources",
    research_document_retrieval: "Find relevant passages · Private documents",
    research_evidence_extractor: "Score evidence relevance",
    research_coverage_evaluator: "Evaluate coverage",
    research_routing: "Decide next step",
    writer_prepare_evidence: "Prepare evidence",
    writer_synthesis: "Synthesize report",
    writer_critic: "Critic review",
    writer_citation_checker: "Verify citations",
    finalize_report: "Finalize report",
};

const PLANNING = ["input_guardrail"];
const RESEARCH = [
    "research_planner", "research_query_generator", "research_source_selection", "research_web_search", "research_source_processor",
    "research_retrieval", "research_document_retrieval", "research_evidence_extractor", "research_coverage_evaluator",
    "research_routing",
];
const WRITING = ["writer_prepare_evidence", "writer_synthesis", "writer_critic", "writer_citation_checker", "finalize_report"];
const EVENT_TYPES = [
    "workflow_started", "stage_started", "stage_completed", "stage_failed",
    "research_retry", "research_stopped", "workflow_completed", "workflow_failed",
];
const TERMINAL = new Set(["completed", "failed"]);
const STORAGE_KEY = "deep-research:last-job";
const OWNER_KEY = "deep-research:owner-id";
const DOCUMENT_SELECTION_ISSUE = "document-selection";
const FALLBACK_POLL_INITIAL_MS = 5000;
const FALLBACK_POLL_MAX_MS = 30000;

let jobId = null;
let jobStartedAt = null;
let jobEndedAt = null;
let stream = null;
let pollTimer = null;
let elapsedTimer = null;
let streamCloseTimer = null;
let events = [];
let highestSequence = 0;
let runToken = 0;
let statusLoading = false;
let resultShown = false;
let terminalEventSeen = false;
let pipelineFailed = false;
let streamAvailable = false;
let pollDelay = FALLBACK_POLL_INITIAL_MS;
let collapsedRounds = new Set();
const expandedStageDetails = new Set();
let previousLatestRound = null;
let documents = [];
let selectedDocumentIds = new Set();

function getOwnerId() {
    try {
        let value = localStorage.getItem(OWNER_KEY);
        if (!value) {
            value = `browser-${crypto.randomUUID()}`;
            localStorage.setItem(OWNER_KEY, value);
        }
        return value;
    } catch {
        return `browser-${Date.now()}`;
    }
}

const ownerId = getOwnerId();

function show(element) { element.classList.remove("hidden"); }
function hide(element) { element.classList.add("hidden"); }
function clear(element) { element.replaceChildren(); }
function label(value) {
    if (value == null || value === "") return "Not available";
    return String(value).replaceAll("_", " ").replace(/\b\w/g, letter => letter.toUpperCase());
}
function create(tag, className = "", content = null) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== null) node.textContent = String(content);
    return node;
}
function renderReportText(parent, value) {
    // Render only inline emphasis/code as DOM nodes. Never interpret report HTML
    // or inject it with innerHTML; the raw Markdown disclosure stays unchanged.
    clear(parent);
    const text = String(value ?? "");
    const tokens = /\\([\\`*])|`([^`\n]+)`|\*\*([\s\S]+?)\*\*|\*([^*\n]+)\*/g;
    let offset = 0;
    for (const match of text.matchAll(tokens)) {
        parent.append(document.createTextNode(text.slice(offset, match.index)));
        if (match[1]) parent.append(document.createTextNode(match[1]));
        else {
            const tag = match[2] !== undefined ? "code" : match[3] !== undefined ? "strong" : "em";
            const content = match[2] ?? match[3] ?? match[4];
            const node = create(tag);
            if (tag === "code") node.textContent = content;
            else renderReportText(node, content);
            parent.append(node);
        }
        offset = match.index + match[0].length;
    }
    parent.append(document.createTextNode(text.slice(offset)));
    return parent;
}
function renderReportMarkdown(parent, value) {
    clear(parent);
    const text = String(value ?? "");
    if (typeof marked === "undefined" || typeof DOMPurify === "undefined") return renderReportText(parent, text);
    // Reports are untrusted content. Parse GFM, then sanitize before insertion;
    // exclude images/embeds so viewing a report cannot fetch remote resources.
    const fragment = DOMPurify.sanitize(marked.parse(text, {gfm: true}), {
        RETURN_DOM_FRAGMENT: true,
        ALLOWED_TAGS: ["p", "br", "hr", "strong", "em", "del", "code", "pre", "blockquote",
            "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6", "a", "table", "thead", "tbody", "tr", "th", "td"],
        ALLOWED_ATTR: ["href", "title", "align", "start"],
        ADD_URI_SAFE_ATTR: ["align", "start"],
        ALLOW_DATA_ATTR: false, ALLOW_ARIA_ATTR: false,
        ALLOWED_URI_REGEXP: /^(?:https?:|#)/i,
    });
    parent.append(fragment);
    parent.querySelectorAll("a").forEach(anchor => {
        anchor.target = "_blank";
        anchor.rel = "noopener noreferrer";
    });
    parent.querySelectorAll("table").forEach(table => {
        const wrapper = create("div", "report-table-scroll");
        wrapper.tabIndex = 0;
        wrapper.setAttribute("role", "region");
        wrapper.setAttribute("aria-label", "Report table; scroll horizontally for more columns");
        table.replaceWith(wrapper);
        wrapper.append(table);
    });
    return parent;
}
function sourceMode() {
    return document.querySelector('input[name="source-mode"]:checked')?.value || "web";
}
function usesDocuments(mode = sourceMode()) { return mode !== "web"; }
function setDocumentStatus(message, type = "neutral", issue = "") {
    ui.documentStatus.textContent = message;
    ui.documentStatus.dataset.type = type;
    if (issue) ui.documentStatus.dataset.issue = issue;
    else delete ui.documentStatus.dataset.issue;
}
function clearDocumentSelectionError() {
    if (ui.documentStatus.dataset.issue === DOCUMENT_SELECTION_ISSUE) {
        setDocumentStatus(`${documents.length} document${documents.length === 1 ? "" : "s"} available.`);
    }
    if (ui.status.dataset.issue === DOCUMENT_SELECTION_ISSUE) {
        hide(ui.status);
        delete ui.status.dataset.issue;
    }
}
function replayAttention(element) {
    element.classList.remove("attention-pulse");
    void element.offsetWidth; // Restart the animation on repeated invalid submissions.
    element.classList.add("attention-pulse");
}
function formatBytes(value) {
    if (!Number.isFinite(value)) return "Unknown size";
    return value < 1024 * 1024
        ? `${Math.max(1, Math.round(value / 1024))} KB`
        : `${(value / (1024 * 1024)).toFixed(1)} MB`;
}
function renderDocuments() {
    clear(ui.documentList);
    if (!documents.length) {
        ui.documentList.append(create("p", "empty-message", "No documents uploaded yet."));
        return;
    }
    for (const item of documents) {
        const row = create("label", "document-item");
        const input = create("input");
        input.type = "checkbox";
        input.value = item.document_id;
        input.checked = selectedDocumentIds.has(item.document_id);
        input.disabled = ui.submit.disabled;
        input.addEventListener("change", () => {
            input.checked
                ? selectedDocumentIds.add(item.document_id)
                : selectedDocumentIds.delete(item.document_id);
            if (selectedDocumentIds.size) clearDocumentSelectionError();
        });
        row.append(
            input,
            create("span", "", item.filename),
            create("small", "", `${formatBytes(item.size_bytes)} · ${label(item.status)}`),
        );
        ui.documentList.append(row);
    }
}
async function loadDocuments() {
    setDocumentStatus("Loading documents…");
    try {
        const response = await fetch("/documents", {
            headers: { "X-Owner-Id": ownerId },
            cache: "no-store",
        });
        documents = await readJson(response);
        const availableIds = new Set(documents.map(item => item.document_id));
        selectedDocumentIds = new Set(
            [...selectedDocumentIds].filter(id => availableIds.has(id))
        );
        renderDocuments();
        setDocumentStatus(`${documents.length} document${documents.length === 1 ? "" : "s"} available.`);
        return true;
    } catch (error) {
        setDocumentStatus(error instanceof Error ? error.message : "Unable to load documents.", "error");
        return false;
    }
}
async function uploadPendingDocuments() {
    const files = [...ui.documentUpload.files];
    if (!files.length) return true;

    setDocumentStatus(`Uploading ${files.length} document${files.length === 1 ? "" : "s"}…`);
    try {
        for (const file of files) {
            const body = new FormData();
            body.append("file", file);
            const response = await fetch("/documents", {
                method: "POST",
                headers: { "X-Owner-Id": ownerId },
                body,
            });
            const uploaded = await readJson(response);
            selectedDocumentIds.add(uploaded.document_id);
        }
        ui.documentUpload.value = "";
        return true;
    } catch (error) {
        const message = error instanceof Error ? error.message : "Document upload failed.";
        setDocumentStatus(message, "error");
        showStatus("Document upload failed", message, "error");
        return false;
    }
}
function showDocumentIssue(title, message) {
    setDocumentStatus(message, "error", DOCUMENT_SELECTION_ISSUE);
    showStatus(title, message, "error", DOCUMENT_SELECTION_ISSUE);
    replayAttention(ui.documentPanel);
    replayAttention(ui.status);
}
async function validateSelectedDocuments() {
    const selectedBeforeRefresh = [...selectedDocumentIds];
    if (!await loadDocuments()) {
        showDocumentIssue(
            "Unable to verify documents",
            "The selected documents could not be verified. Refresh the list and try again.",
        );
        return false;
    }

    const missingSelection = selectedBeforeRefresh.some(
        documentId => !selectedDocumentIds.has(documentId)
    );
    if (missingSelection) {
        showDocumentIssue(
            "Documents unavailable",
            "One or more selected documents are no longer available. Select or upload them again.",
        );
        return false;
    }
    return true;
}
function updateSourceControls() {
    const mode = sourceMode();
    ui.sourceIndicator.textContent = ({ web: "Web research", documents: "Document research", hybrid: "Hybrid research" })[mode];
    if (!usesDocuments(mode)) clearDocumentSelectionError();
    usesDocuments(mode) ? show(ui.documentPanel) : hide(ui.documentPanel);
    if (usesDocuments(mode) && !documents.length) void loadDocuments();
}
function setBusy(busy) {
    ui.submit.disabled = busy;
    ui.question.disabled = busy;
    ui.documentUpload.disabled = busy;
    ui.refreshDocuments.disabled = busy;
    document.querySelectorAll('input[name="source-mode"], .document-item input').forEach(input => { input.disabled = busy; });
    document.querySelectorAll(".example-chip").forEach(button => { button.disabled = busy; });
    ui.submit.querySelector("span").textContent = busy ? "Research in progress" : "Start research";
    busy ? show(ui.spinner) : hide(ui.spinner);
    busy ? hide(ui.statusIcon) : show(ui.statusIcon);
}
function showStatus(title, message, type = "neutral", issue = "") {
    ui.statusTitle.textContent = title;
    ui.statusMessage.textContent = message;
    ui.status.dataset.type = type;
    ui.status.setAttribute("role", type === "error" ? "alert" : "status");
    ui.status.setAttribute("aria-live", type === "error" ? "assertive" : "polite");
    if (issue) ui.status.dataset.issue = issue;
    else delete ui.status.dataset.issue;
    show(ui.status);
}
function setConnection(text = "") {
    if (text) { ui.connection.textContent = text; show(ui.connection); }
    else hide(ui.connection);
}
function resetReport() {
    hide(ui.result);
    ui.reportHeading.textContent = "Research report";
    hide(ui.errorsSection);
    ui.errorsSection.open = false;
    ui.markdownSection.open = false;
    ui.summary.textContent = "";
    ui.markdown.textContent = "";
    [ui.findings, ui.limitations, ui.unverified, ui.sources, ui.errors, ui.metadata].forEach(clear);
    hide(ui.unverifiedSection);
}
function resetPipeline() {
    events = [];
    highestSequence = 0;
    resultShown = false;
    terminalEventSeen = false;
    pipelineFailed = false;
    collapsedRounds = new Set();
    expandedStageDetails.clear();
    previousLatestRound = null;
    jobStartedAt = Date.now();
    jobEndedAt = null;
    ui.coverageStatus.textContent = "Awaiting evaluation";
    ui.coverageDescription.textContent = "The evidence evaluator will check research coverage after scoring.";
    ui.coverageGapCount.textContent = "—";
    ui.coverageAssessmentCount.textContent = "—";
    renderPipeline();
    show(ui.pipeline);
    updateElapsed();
}
function stopNetwork() {
    if (stream) { stream.close(); stream = null; }
    clearTimeout(pollTimer); pollTimer = null;
    clearInterval(elapsedTimer); elapsedTimer = null;
    clearTimeout(streamCloseTimer); streamCloseTimer = null;
    streamAvailable = false;
    pollDelay = FALLBACK_POLL_INITIAL_MS;
}
function setJobUrl(id) {
    const url = new URL(window.location.href);
    url.searchParams.set("job", id);
    window.history.replaceState({}, "", url);
    try { localStorage.setItem(STORAGE_KEY, id); } catch { /* Storage might be blocked. */ }
}
function clearSavedJob() {
    try { localStorage.removeItem(STORAGE_KEY); } catch { /* Ignore storage restrictions. */ }
    const url = new URL(window.location.href);
    url.searchParams.delete("job");
    window.history.replaceState({}, "", url);
}
function parseTimestamp(raw) {
    if (!raw) return null;
    // SQLite CURRENT_TIMESTAMP uses UTC without a timezone suffix.
    const normalized = /\d{4}-\d\d-\d\d \d\d:\d\d:\d\d/.test(raw)
        ? `${raw.replace(" ", "T")}Z` : raw;
    const date = new Date(normalized);
    return Number.isNaN(date.getTime()) ? null : date.getTime();
}
function updateElapsed() {
    const seconds = Math.max(0, Math.floor(((jobEndedAt || Date.now()) - (jobStartedAt || Date.now())) / 1000));
    const minutes = Math.floor(seconds / 60);
    ui.elapsed.textContent = `${String(minutes).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")} elapsed`;
}
function getStageState(stage, round = null) {
    // Two UI rows describe the same graph retrieval node, not extra executions.
    if (stage === "research_document_retrieval") {
        const selection = events.find(event => event.stage === "research_source_selection"
            && event.iteration === round && event.event_type === "stage_completed");
        if (selection?.details?.document_count === 0 || selection?.details?.source_mode === "web") return "skipped";
        stage = "research_retrieval";
    }
    let state = "pending";
    const stageRound = round ?? (stage === "research_planner" ? 1 : null);
    for (const event of events) {
        if (event.stage !== stage) continue;
        if (stageRound !== null && event.iteration !== stageRound) continue;
        if (event.event_type === "stage_started") state = "running";
        if (event.event_type === "stage_completed") state = "completed";
        if (event.event_type === "stage_failed") state = "failed";
    }
    return state === "running" && pipelineFailed ? "failed" : state;
}
function appendStageMetrics(parent, metrics) {
    const grid = create("div", "stage-metrics");
    for (const [name, value] of metrics) {
        if (!Number.isFinite(value)) continue;
        const metric = create("div");
        metric.append(create("strong", "", value), create("span", "", name));
        grid.append(metric);
    }
    if (grid.children.length) parent.append(grid);
}
function renderStageDetails(stage, round, selected = null, keySuffix = "") {
    if (stage === "research_document_retrieval" && getStageState(stage, round) === "skipped") {
        return create("p", "stage-skip-note", "No private documents attached. Document retrieval was skipped.");
    }
    const stageRound = round ?? (stage === "research_planner" ? 1 : null);
    const eventStage = stage === "research_document_retrieval" ? "research_retrieval" : stage;
    // Writer stages can run twice (draft and revision), so callers pass the run's own event.
    const event = selected ?? events.filter(item => item.stage === eventStage
        && ["stage_completed", "stage_failed"].includes(item.event_type)
        && (stageRound === null || item.iteration === stageRound)).at(-1);
    if (!event) return null;
    const data = event.details || {};
    const body = create("div", "stage-output-body");
    let caption = "View details";

    if (stage === "research_planner") {
        if (!Number.isInteger(data.active_subquestion_count)) return null;
        caption = `${data.active_subquestion_count} subquestions · View plan`;
        const list = create("ol", "stage-detail-list");
        (data.subquestions || []).forEach(task => {
            const item = create("li");
            item.append(create("strong", "detail-question", task.question || "Research task"));
            if (task.objective) item.append(create("p", "", task.objective));
            if (task.parent_question) item.append(create("small", "detail-muted", `Follow-up to: ${task.parent_question}`));
            list.append(item);
        });
        body.append(list);
        if (!list.children.length) body.append(create("p", "detail-muted",
            Array.isArray(data.subquestions) ? "No new subquestions were generated." : "Only counts were recorded for this event."));
    } else if (stage === "research_query_generator") {
        if (!Number.isInteger(data.query_count)) return null;
        caption = `${data.query_count} queries · View by subquestion`;
        (data.query_groups || []).forEach(group => {
            const section = create("section", "stage-detail-group");
            section.append(create("strong", "detail-question", group.question || "Research task"));
            const list = create("ol", "stage-detail-list");
            (group.queries || []).forEach(query => {
                const item = create("li");
                item.append(create("p", "query-text", query.query));
                if (query.purpose) item.append(create("small", "detail-muted", query.purpose));
                list.append(item);
            });
            section.append(list);
            body.append(section);
        });
        if (!data.query_count) body.append(create("p", "detail-muted", "No new queries were accepted for this round."));
    } else if (stage === "research_web_search") {
        if (!Number.isInteger(data.result_count)) return null;
        body.append(create("p", "detail-muted", `${event.job_id ? `Job ${event.job_id} · ` : ""}Round ${event.iteration ?? stageRound} only${Number.isInteger(data.executed_query_count) ? ` · ${data.executed_query_count} queries executed` : ""}. Not a cumulative job total.`));
        caption = `${data.discovered_count ?? data.result_count} ${Number.isInteger(data.discovered_count) ? "discovered" : "retained"} results · View providers`;
        const names = {tavily: "Tavily", brave: "Brave", wikipedia: "Wikipedia", arxiv: "arXiv"};
        appendStageMetrics(body, Object.entries(data.discovered_provider_counts || data.provider_counts || {}).map(([provider, count]) => [names[provider] || label(provider), count]));
        if (Number.isInteger(data.discovered_count)) body.append(create("p", "detail-muted", `Provider counts show discovery before deduplication, aggregate ranking, and score filtering. Zero means no results returned, not that the provider was omitted.`));
        body.append(create("p", "detail-muted", `${data.result_count} retained search results after filtering; repeated sources across queries may be counted more than once.`));
        if (data.discovered_provider_counts) body.append(create("p", "detail-muted", `Retained by provider: ${Object.entries(data.provider_counts || {}).map(([name, count]) => `${names[name] || label(name)} ${count}`).join(" · ") || "None"}.`));
        if (data.failed_query_count) body.append(create("p", "detail-warning", `${data.failed_query_count} queries failed or timed out.`));
        if (data.failed_branches?.length) body.append(create("p", "detail-warning", `Provider branches failed: ${data.failed_branches.join(", ")}. Other results were retained.`));
        if (data.web_fallback_activated) body.append(create("p", "detail-muted", "Brave fallback was activated."));
    } else if (stage === "research_source_processor") {
        if (!Number.isInteger(data.source_count)) return null;
        const count = data.extracted_count ?? data.source_count;
        caption = `${count} pages ${Number.isInteger(data.extracted_count) ? "extracted" : "available"} · View sources`;
        const counts = data.extracted_provider_counts || data.provider_counts || {};
        appendStageMetrics(body, [["Total", count], ["Tavily", counts.tavily || 0],
            ["Wikipedia", counts.wikipedia || 0], ["arXiv", counts.arxiv || 0],
            ...(counts.brave ? [["Brave", counts.brave]] : [])]);
        appendStageMetrics(body, [["Extraction attempts", data.attempted_count], ["Reused pages", data.reused_count], ["Budget deferred", data.deferred_count]]);
        body.append(create("p", "detail-muted", `${data.source_count} unique pages available for this round, including reused content. Provider labels identify discovery, not the extraction service. Failed or empty extractions are not counted as extracted pages.`));
        if (data.extraction_fallback_count) body.append(create("p", "detail-muted", `Firecrawl fallback used for ${data.extraction_fallback_count} available pages.`));
        (data.source_quality || []).forEach(source => {
            const pages = Number.isInteger(source.pages_processed) && Number.isInteger(source.total_pages)
                ? ` · ${source.pages_processed}/${source.total_pages} PDF pages processed` : "";
            const warnings = [source.pdf_truncated ? "PDF page limit reached" : "", source.content_truncated ? "text character limit reached" : ""].filter(Boolean);
            body.append(create("p", warnings.length || source.content_source === "abstract" ? "detail-warning" : "detail-muted",
                `${source.title}: ${source.content_source === "abstract" ? "Abstract only" : label(source.content_source)}${pages}${warnings.length ? ` · ${warnings.join("; ")}` : ""}`));
        });
    } else if (["research_retrieval", "research_document_retrieval"].includes(stage)) {
        const prefix = stage === "research_document_retrieval" ? "document" : "web";
        const count = data[`${prefix}_chunk_count`];
        if (!Number.isInteger(count)) return null;
        caption = `${count} passages selected · View by query`;
        body.append(create("p", "detail-muted", "Hybrid retrieval matches before evidence scoring. Counts are per subquestion; the same passage can match multiple queries. Query counts overlap and must not be added together."));
        const groups = data[`${prefix}_groups`] || [];
        groups.forEach(group => {
            const section = create("section", "stage-detail-group");
            section.append(create("strong", "detail-question", group.question || "Research task"));
            section.append(create("p", "detail-muted", `${group.passage_count} distinct passages selected for this task · ${group.estimated_tokens} context tokens`));
            const list = create("ul", "stage-detail-list");
            (group.query_matches || []).forEach(match => {
                const item = create("li");
                item.append(create("p", "query-text", match.query), create("small", "detail-muted", `${match.retrieved_count} retrieved · ${match.selected_count} used (others were already included by another search, over the per-source cap, or over the token budget)`));
                list.append(item);
            });
            section.append(list);
            body.append(section);
        });
        if (!groups.length) body.append(create("p", "detail-muted", count ? "Query details were not recorded for this event." : "No passages were retrieved from these sources."));
        if (prefix === "document") body.append(create("p", "detail-muted", "Uses the job's private index in the same retrieval step. Document contents and private file metadata are not included in progress events."));
    } else if (stage === "research_evidence_extractor") {
        if (!Number.isInteger(data.accepted_chunk_count)) return null;
        caption = `${data.accepted_chunk_count} passages kept · View scoring`;
        appendStageMetrics(body, [
            ["Scored", data.scored_chunk_count], ["Kept this round", data.accepted_chunk_count],
            ["Below threshold", data.low_relevance_chunk_count], ["Capacity skipped", data.capacity_skipped_chunk_count],
        ]);
        if (data.jev_batch_count && Number.isFinite(data.jev_threshold)) body.append(create("p", "detail-muted", `Scored by Jev for ${data.jev_batch_count} ${data.jev_batch_count === 1 ? "task" : "tasks"}: passages need a usefulness probability of at least ${data.jev_threshold.toFixed(2)}.`));
        if (Number.isFinite(data.min_evidence_relevance) && (data.llm_batch_count || !data.jev_batch_count)) body.append(create("p", "detail-muted", `${data.jev_batch_count ? `Scored by the LLM for ${data.llm_batch_count} ${data.llm_batch_count === 1 ? "task" : "tasks"}. ` : ""}Minimum relevance: ${data.min_evidence_relevance.toFixed(2)}. Below-threshold passages are filtered, not necessarily factually incorrect.`));
        body.append(create("p", "detail-muted", `Total evidence retained across rounds: ${data.total_evidence_count ?? "Not recorded"}. Previously accepted passages are not rescored.`));
        const invalid = (data.missing_chunk_count || 0) + (data.unknown_chunk_count || 0) + (data.duplicate_chunk_count || 0);
        if (invalid) body.append(create("p", "detail-warning", `Scoring response issues: ${data.missing_chunk_count || 0} missing, ${data.unknown_chunk_count || 0} unknown IDs, ${data.duplicate_chunk_count || 0} duplicate judgments. These decisions were not accepted.`));
        if (data.failed_batch_count) body.append(create("p", "detail-warning", `${data.failed_batch_count} scoring batches failed; those passages were not scored.`));
        if (data.capacity_reached) body.append(create("p", "detail-muted", "The evidence capacity was reached. Later batches may not have been scored."));
        (data.scoring_groups || []).forEach(group => {
            const section = create("section", "stage-detail-group");
            section.append(create("strong", "detail-question", group.question || "Research task"));
            (group.passages || []).forEach(passage => {
                const card = create("details", "passage-audit");
                card.dataset.decision = passage.decision;
                const score = Number.isFinite(passage.relevance_score) ? passage.relevance_score.toFixed(2) : "Not scored";
                const decision = passage.decision === "below_threshold" ? "Rejected (below threshold)" : label(passage.decision);
                card.append(create("summary", "", `${decision} · ${score} · ${passage.title || "Source"}`));
                card.append(create("p", "detail-muted", passage.chunk_id || "Passage"));
                if (passage.queries?.length) card.append(create("p", "query-text", `Matched queries: ${passage.queries.join("; ")}`));
                if (passage.source_type === "document") card.append(create("p", "detail-muted", "Private passage text and file details are not stored in progress events."));
                else {
                    card.append(create("p", "passage-preview", passage.preview || "No preview recorded."));
                    if (passage.preview_truncated) card.append(create("small", "detail-muted", "Preview shortened to 800 characters; scoring used the full passage."));
                    const url = safeHttpUrl(passage.uri);
                    if (url) {
                        const link = create("a", "", "Open source");
                        link.href = url; link.target = "_blank"; link.rel = "noopener noreferrer";
                        card.append(link);
                    }
                }
                section.append(card);
            });
            body.append(section);
        });
    } else if (stage === "research_coverage_evaluator") {
        if (!data.coverage_status) return null;
        const ready = data.coverage_status === "sufficient";
        caption = `${ready ? "Ready to write" : "More evidence needed"} · View assessments`;
        appendStageMetrics(body, Object.entries(data.assessment_counts || {}).map(([status, count]) => [label(status), count]));
        body.append(create("p", "detail-muted", ready
            ? "The core request has enough evidence. Remaining gaps are report limitations, not reasons for another round."
            : "Material gaps still prevent a grounded answer to the core request."));
        (data.assessments || []).forEach(assessment => {
            const section = create("section", "stage-detail-group");
            const badge = create("span", "assessment-badge", label(assessment.status));
            badge.dataset.status = assessment.status;
            section.append(badge, create("strong", "detail-question", assessment.question || "Research objective"));
            if (assessment.reason) section.append(create("p", "", assessment.reason));
            section.append(create("small", "detail-muted", `${assessment.supporting_evidence_count || 0} supporting passages · Evidence strength: ${label(assessment.evidence_strength)}`));
            body.append(section);
        });
        if (data.gaps?.length) {
            body.append(create("strong", "detail-question", ready ? "Remaining limitations" : "Material gaps"));
            const list = create("ul", "stage-detail-list");
            data.gaps.forEach(gap => {
                const item = create("li");
                if (gap.question) item.append(create("strong", "detail-question", gap.question));
                item.append(create("p", "", gap.description));
                const kind = gap.blocking ? "Blocking: triggers more research" : "Limitation: reported, no more research";
                item.append(create("small", "detail-muted", `${kind}${Number.isInteger(gap.priority) ? ` · Priority ${gap.priority}/5` : ""}`));
                list.append(item);
            });
            body.append(list);
        }
    } else if (stage === "writer_prepare_evidence") {
        if (!Number.isInteger(data.evidence_count)) return null;
        caption = `${data.evidence_count} evidence passages · View by subquestion`;
        const lineage = data.source_lineage || {};
        appendStageMetrics(body, [
            ["Evidence passages", data.evidence_count], ["Sources supplied", lineage.supplied_to_writer], ["Sources accepted", lineage.accepted],
        ]);
        body.append(create("p", "detail-muted", "Every accepted passage goes to the writer, grouped by subquestion with the strongest first."));
        const list = create("ul", "stage-detail-list");
        (data.evidence_groups || []).forEach(group => {
            const item = create("li");
            item.append(create("strong", "detail-question", group.question || "Research task"),
                create("small", "detail-muted", `${group.evidence_count} passages · ${group.source_count} sources`));
            list.append(item);
        });
        if (list.children.length) body.append(list);
    } else if (stage === "writer_synthesis") {
        if (!Number.isInteger(data.finding_count)) return null;
        caption = `${data.revision ? "Revised draft" : "Draft"} · ${data.finding_count} findings · View outline`;
        appendStageMetrics(body, [
            ["Findings", data.finding_count], ["Evidence supplied", data.evidence_count], ["Limitations", data.limitation_count],
        ]);
        if (data.revision) body.append(create("p", "detail-muted", "Rewritten once from the critic's feedback, using the same evidence."));
        if (data.title) body.append(create("strong", "detail-question", data.title));
        const sections = new Map();
        (data.findings || []).forEach(finding => {
            const heading = finding.section || "Detailed Analysis";
            if (!sections.has(heading)) {
                const section = create("section", "stage-detail-group");
                section.append(create("strong", "detail-question", heading), create("ul", "stage-detail-list"));
                sections.set(heading, section);
                body.append(section);
            }
            const item = create("li");
            item.append(create("p", "", finding.title || "Finding"),
                create("small", "detail-muted", `${finding.citation_count} cited evidence passages`));
            sections.get(heading).children[1].append(item);
        });
    } else if (stage === "writer_critic") {
        if (!data.verdict) return null;
        const revise = data.verdict === "revise";
        caption = revise ? `Revision requested · ${data.issue_count} issues · View feedback` : "Approved · View review";
        body.append(create("p", revise ? "detail-warning" : "detail-muted", revise
            ? "The draft is revised once with the same evidence. The job does not return to research."
            : "The draft is grounded and complete enough to go to citation checking."));
        appendStageMetrics(body, Object.entries(data.issue_types || {}).map(([type, count]) => [label(type), count]));
        if (data.issues?.length) {
            const list = create("ul", "stage-detail-list");
            data.issues.forEach(issue => {
                const item = create("li");
                const target = Number.isInteger(issue.finding_index) ? `Finding ${issue.finding_index + 1}` : "Whole report";
                item.append(create("strong", "detail-question", `${label(issue.issue_type)} · ${target}`), create("p", "", issue.instruction || ""));
                list.append(item);
            });
            body.append(list);
        } else if (data.issue_count) {
            body.append(create("p", "detail-muted", "Feedback text is not included in progress events for jobs with private documents."));
        }
    } else if (stage === "writer_citation_checker") {
        if (!Number.isInteger(data.verified_finding_count)) return null;
        caption = `${data.verified_finding_count} verified · ${data.unverified_item_count} unverified · View checks`;
        appendStageMetrics(body, [
            ["Verified findings", data.verified_finding_count], ["Moved to unverified", data.unverified_item_count], ["Issues", data.issue_count],
        ]);
        if (data.semantic_check_unavailable) body.append(create("p", "detail-warning", "Semantic verification was unavailable; citations were checked for valid references only."));
        [["Issues found", data.issues], ["Moved to 'Additional material (not citation-verified)'", data.unverified_titles]].forEach(([title, items]) => {
            if (!items?.length) return;
            body.append(create("strong", "detail-question", title));
            const list = create("ul", "stage-detail-list");
            items.forEach(text => list.append(create("li", "", text)));
            body.append(list);
        });
    } else if (stage === "finalize_report") {
        const lineage = Object.entries(data.source_lineage || {});
        if (!lineage.length) return null;
        caption = "View source lineage";
        appendStageMetrics(body, lineage.map(([name, count]) => [label(name), count]));
        body.append(create("p", "detail-muted", "Sources collected → retrieved → accepted as evidence → supplied to the writer → cited in the final report."));
    } else return null;

    if (data.reason) body.append(create("p", "detail-muted", data.reason));
    if (data.details_truncated) body.append(create("p", "detail-muted", "Showing the first 100 items. Totals include all items; long text is shortened with an ellipsis."));
    const output = create("details", "stage-output");
    const key = `${stage}:${stageRound ?? "workflow"}${keySuffix}`;
    output.open = expandedStageDetails.has(key);
    output.addEventListener("toggle", () => {
        if (!output.isConnected) return;
        output.open ? expandedStageDetails.add(key) : expandedStageDetails.delete(key);
    });
    output.append(create("summary", "", caption), body);
    return output;
}
function stageRow(name, state, details) {
    const row = create("li", "stage-row");
    row.dataset.state = state;
    const heading = create("div", "stage-heading");
    heading.append(
        create("span", "stage-node", ({ completed: "✓", failed: "×", running: "◆", pending: "·", skipped: "↷" })[state]),
        create("span", "stage-name", name),
        create("span", "stage-caption", ({ completed: "Done", failed: "Failed", running: "Running", pending: "Waiting", skipped: "Skipped" })[state]),
    );
    row.append(heading);
    if (details) row.append(details);
    return row;
}
function renderStages(stages, round = null) {
    const list = create("ol", "stage-list");
    for (const stage of stages) {
        list.append(stageRow(STAGES[stage], getStageState(stage, round), renderStageDetails(stage, round)));
    }
    return list;
}
function writerRuns() {
    // Writer stages run in order, but synthesis can run twice; keep each run separately.
    const runs = [];
    for (const event of events) {
        if (!WRITING.includes(event.stage)) continue;
        if (event.event_type === "stage_started") {
            runs.push({ stage: event.stage, state: "running", event: null });
        } else if (["stage_completed", "stage_failed"].includes(event.event_type)) {
            let run = [...runs].reverse().find(item => item.stage === event.stage && !item.event);
            if (!run) { run = { stage: event.stage }; runs.push(run); }
            run.event = event;
            run.state = event.event_type === "stage_completed" ? "completed" : "failed";
        }
    }
    return runs;
}
function renderWriterGroup() {
    const group = create("section", "pipeline-group");
    const head = create("div", "group-head");
    head.append(create("h3", "", "03 · Write & verify"), create("span", "group-hint",
        withDuration("Draft, review, revise once if needed, verify", stageDuration(event => WRITING.includes(event.stage)))));
    const list = create("ol", "stage-list");
    const runs = writerRuns();
    let synthesisRuns = 0;
    runs.forEach((run, index) => {
        const revision = run.stage === "writer_synthesis" && synthesisRuns++ > 0;
        const state = run.state === "running" && pipelineFailed ? "failed" : run.state;
        const name = revision ? "Revise report (critic feedback)" : STAGES[run.stage];
        list.append(stageRow(name, state, run.event ? renderStageDetails(run.stage, null, run.event, `:${index}`) : null));
        if (run.stage === "writer_critic" && run.event?.details?.verdict === "revise") {
            const note = create("li", "round-note");
            note.dataset.kind = "retry";
            note.append(create("strong", "", "↻ Revision requested"),
                create("p", "", `${run.event.details.issue_count ?? "Some"} issues to fix with the same evidence; at most one revision.`));
            list.append(note);
        }
    });
    // Stages not reached yet; the critic does not review a revised draft again.
    const last = runs.length ? WRITING.indexOf(runs.at(-1).stage) : -1;
    WRITING.slice(last + 1)
        .filter(stage => !(stage === "writer_critic" && synthesisRuns > 1))
        .forEach(stage => list.append(stageRow(STAGES[stage], "pending", null)));
    group.append(head, list);
    return group;
}
function formatDuration(ms) {
    const seconds = Math.max(0, Math.round(ms / 1000));
    const minutes = Math.floor(seconds / 60);
    return minutes ? `${minutes} min ${seconds % 60} sec` : `${seconds} sec`;
}
function stageDuration(belongs) {
    // Time from the section's first event to its last; "so far" while a stage is still running.
    const own = events.filter(belongs);
    const times = own.map(event => parseTimestamp(event.timestamp)).filter(Number.isFinite);
    if (times.length < 2) return "";
    const open = own.at(-1)?.event_type === "stage_started" && !pipelineFailed;
    return `${formatDuration(Math.max(...times) - Math.min(...times))}${open ? " so far" : ""}`;
}
function withDuration(hint, duration) {
    return duration ? `${hint} · ${duration}` : hint;
}
function renderGroup(title, hint, stages) {
    const group = create("section", "pipeline-group");
    const head = create("div", "group-head");
    head.append(create("h3", "", title), create("span", "group-hint", withDuration(hint, stageDuration(event => stages.includes(event.stage)))));
    group.append(head, renderStages(stages));
    return group;
}
function researchRounds() {
    return [...new Set(events.filter(event =>
        RESEARCH.includes(event.stage)
        && Number.isInteger(event.iteration))
        .map(event => event.iteration))].sort((a, b) => a - b);
}
function roundEvent(round, type) {
    return events.filter(event => event.iteration === round && event.event_type === type).at(-1);
}
function roundCoverage(round) {
    return events.filter(event => event.iteration === round &&
        event.stage === "research_coverage_evaluator" && event.event_type === "stage_completed").at(-1);
}
function researchStages(round) {
    const stages = RESEARCH;
    const selection = events.find(event =>
        event.iteration === round
        && event.stage === "research_source_selection"
        && event.event_type === "stage_completed");
    if (selection?.details?.source_mode === "documents") {
        return stages.filter(stage => !["research_web_search", "research_source_processor", "research_retrieval"].includes(stage));
    }
    return stages;
}
function buildRoundCard(round, latest) {
    const retry = roundEvent(round, "research_retry");
    const stop = roundEvent(round, "research_stopped");
    const coverage = roundCoverage(round);
    const details = create("details", "round-card");
    details.open = !collapsedRounds.has(round) && round === latest;
    // Preserve explicit user expansion across subsequent SSE updates.
    if (collapsedRounds.has(`expanded:${round}`)) details.open = true;
    details.addEventListener("toggle", () => {
        if (details.open) { collapsedRounds.delete(round); collapsedRounds.add(`expanded:${round}`); }
        else { collapsedRounds.add(round); collapsedRounds.delete(`expanded:${round}`); }
    });
    const summary = create("summary");
    const left = create("span", "round-summary-text");
    left.append(create("span", "round-indicator", round), create("span", "", `Research round ${round}`));
    const status = retry ? "Follow-up required" : stop ? "Concluded"
        : getStageState("research_routing", round) === "completed" ? "Evaluated" : "In progress";
    const duration = stageDuration(event => RESEARCH.includes(event.stage) && event.iteration === round);
    summary.append(left, create("span", "round-status", withDuration(status, duration)));
    details.append(summary);
    const body = create("div", "round-details");
    body.append(renderStages(researchStages(round), round));
    const decision = retry || stop;
    if (decision) {
        const note = create("div", "round-note");
        note.dataset.kind = retry ? "retry" : "done";
        note.append(create("strong", "", retry ? `↻ Continuing to round ${decision.details?.next_iteration || round + 1}` : "Research pass concluded"));
        const gaps = decision.details?.gap_count;
        const gapLabel = !retry && coverage?.details?.coverage_status === "sufficient" ? "limitation" : "gap";
        const reason = decision.details?.reason || (retry ? "Further evidence is needed." : "Research stage finished.");
        note.append(create("p", "", `${reason}${typeof gaps === "number" ? ` · ${gaps} remaining ${gapLabel}${gaps === 1 ? "" : "s"}` : ""}`));
        body.append(note);
    } else if (coverage?.details?.coverage_status) {
        const note = create("div", "round-note");
        const gapCount = coverage.details.gap_count;
        note.append(create("strong", "", `Coverage: ${label(coverage.details.coverage_status)}`));
        const gapLabel = coverage.details.coverage_status === "sufficient" ? "limitation" : "material gap";
        if (Number.isFinite(gapCount)) note.append(create("p", "", `${gapCount} remaining ${gapLabel}${gapCount === 1 ? "" : "s"}`));
        body.append(note);
    }
    details.append(body);
    return details;
}
function renderPipeline() {
    clear(ui.timeline);
    ui.pipelineJob.textContent = jobId ? `Job ${jobId}` : "Job pending";
    ui.timeline.append(renderGroup("01 · Validate", "Check the request", PLANNING));

    const research = create("section", "pipeline-group");
    const head = create("div", "group-head");
    head.append(create("h3", "", "02 · Research & evaluate"), create("span", "group-hint",
        withDuration("Evidence-driven iterations", stageDuration(event => RESEARCH.includes(event.stage)))));
    research.append(head);
    const rounds = researchRounds();
    if (rounds.length) {
        const latest = rounds.at(-1);
        if (previousLatestRound !== null && previousLatestRound !== latest) {
            collapsedRounds.add(previousLatestRound);
            collapsedRounds.delete(`expanded:${previousLatestRound}`);
        }
        previousLatestRound = latest;
        const stack = create("div", "rounds");
        rounds.forEach(round => stack.append(buildRoundCard(round, latest)));
        research.append(stack);
    } else {
        research.append(create("p", "empty-rounds", "Research rounds will appear when planning begins."));
    }
    ui.timeline.append(research, renderWriterGroup());
    renderInsights();
    renderActivity();
}
function renderInsights() {
    const coverage = [...events].reverse().find(event =>
        event.event_type === "stage_completed" && event.stage === "research_coverage_evaluator"
        && event.details?.coverage_status);
    if (!coverage) return;
    const status = coverage.details.coverage_status;
    ui.coverageStatus.textContent = status === "sufficient" ? "Ready to write"
        : status === "insufficient" ? "More evidence needed" : label(status);
    const gaps = coverage.details.gap_count;
    const assessments = coverage.details.assessment_count;
    ui.coverageGapCount.textContent = Number.isInteger(gaps) ? gaps : "—";
    ui.coverageAssessmentCount.textContent = Number.isInteger(assessments) ? assessments : "—";
    ui.coverageDescription.textContent = Number.isInteger(gaps)
        ? status === "sufficient"
            ? `Round ${coverage.iteration}: enough evidence for a report; ${gaps} remaining limitation${gaps === 1 ? "" : "s"}.`
            : `Round ${coverage.iteration}: ${gaps} material research gap${gaps === 1 ? "" : "s"} identified.`
        : `Round ${coverage.iteration}: evidence evaluation completed.`;
}
function activityDescription(event) {
    const stage = STAGES[event.stage] || label(event.stage);
    if (event.event_type === "workflow_started") return ["Research started", "Workflow initialized"];
    if (event.event_type === "workflow_completed") return ["Research completed", "Final report ready"];
    if (event.event_type === "workflow_failed") return ["Research failed", event.details?.reason || "Execution stopped"];
    if (event.event_type === "research_retry") return [
        `Follow-up research · round ${event.details?.next_iteration || (event.iteration || 0) + 1}`,
        `${event.details?.gap_count ?? "Some"} unresolved research gaps`,
    ];
    if (event.event_type === "research_stopped") return ["Research concluded", event.details?.reason || "Moving to report synthesis"];
    if (event.event_type === "stage_failed") return [`${stage} failed`, event.details?.reason || "Stage error"];
    if (event.event_type === "stage_started") return [`${stage} started`, event.iteration ? `Round ${event.iteration}` : "Workflow"];
    if (event.event_type === "stage_completed" && event.stage === "research_planner") {
        return ["Research plan ready", `${event.details?.active_subquestion_count ?? 0} subquestions · Round ${event.iteration}`];
    }
    if (event.event_type === "stage_completed" && event.stage === "research_query_generator" && Number.isInteger(event.details?.query_count)) {
        return ["Search queries generated", `${event.details.query_count} queries grouped by subquestion · Round ${event.iteration}`];
    }
    if (event.event_type === "stage_completed" && event.stage === "research_evidence_extractor" && Number.isInteger(event.details?.accepted_chunk_count)) {
        return ["Evidence relevance scored", `${event.details.accepted_chunk_count} kept · ${event.details.low_relevance_chunk_count || 0} below threshold · Round ${event.iteration}`];
    }
    if (event.event_type === "stage_completed" && event.stage === "research_coverage_evaluator") {
        return ["Coverage evaluated", `${event.details?.assessment_counts?.supported ?? "—"} supported · ${event.details?.gap_count ?? "—"} gaps / limitations · Round ${event.iteration}`];
    }
    if (event.event_type === "stage_completed" && event.stage === "research_source_selection") {
        const count = event.details?.document_count || 0;
        return ["Sources prepared", count ? `${count} private document${count === 1 ? "" : "s"} ready` : "Web sources selected"];
    }
    if (event.event_type === "stage_completed" && event.stage === "research_web_search") {
        const count = event.details?.discovered_count ?? event.details?.result_count ?? 0;
        const providers = Object.entries(event.details?.discovered_provider_counts || event.details?.provider_counts || {})
            .map(([provider, total]) => `${label(provider)} ${total}`).join(" · ");
        const notes = [`${count} ${Number.isInteger(event.details?.discovered_count) ? "discovered " : ""}result${count === 1 ? "" : "s"}`, providers];
        if (event.details?.web_fallback_activated) notes.push("Brave fallback used");
        const failures = event.details?.partial_failure_count || 0;
        if (failures) notes.push(`${failures} provider branch${failures === 1 ? "" : "es"} unavailable`);
        return ["Public search completed", notes.filter(Boolean).join(" · ")];
    }
    if (event.event_type === "stage_completed" && event.stage === "research_source_processor") {
        const count = event.details?.source_count || 0;
        const fallback = event.details?.extraction_fallback_count || 0;
        return ["Public content extracted", `${event.details?.extracted_count ?? count} pages extracted · ${count} available${fallback ? ` · Firecrawl fallback ${fallback}` : ""}`];
    }
    if (event.event_type === "stage_completed" && event.stage === "research_retrieval") {
        const web = event.details?.web_chunk_count || 0;
        const docs = event.details?.document_chunk_count || 0;
        return ["Relevant passages retrieved", `${web} web · ${docs} document`];
    }
    if (event.event_type === "stage_completed") return [`${stage} completed`, event.iteration ? `Round ${event.iteration}` : "Workflow"];
    return [label(event.event_type), stage];
}
function renderActivity() {
    const visible = events.filter(event => event.event_type !== "stage_started").slice(-7).reverse();
    clear(ui.activity);
    ui.eventCount.textContent = `${events.length} event${events.length === 1 ? "" : "s"}`;
    visible.forEach(event => {
        const item = create("li", "activity-item");
        item.dataset.kind = event.event_type;
        const [title, detail] = activityDescription(event);
        item.append(create("strong", "", title), create("small", "", detail));
        ui.activity.append(item);
    });
    visible.length ? hide(ui.activityEmpty) : show(ui.activityEmpty);
}
function ingestEvent(event) {
    if (!event || event.job_id !== jobId || !Number.isInteger(event.sequence) || event.sequence <= highestSequence) return;
    highestSequence = event.sequence;
    if (event.event_type === "workflow_failed") pipelineFailed = true;
    events.push(event);
    renderPipeline();

    if (!resultShown && event.event_type === "stage_started") {
        showStatus("Research in progress", `${STAGES[event.stage] || "Research"} is running${event.iteration ? ` · round ${event.iteration}` : ""}.`, "loading");
    } else if (!resultShown && event.event_type === "research_retry") {
        showStatus("Another research pass", `Coverage evaluation found gaps. Starting round ${event.details?.next_iteration || (event.iteration || 0) + 1}.`, "loading");
    } else if (!resultShown && event.event_type === "research_stopped") {
        showStatus("Writing the report", "Research has concluded. Evidence synthesis and citation checking are next.", "loading");
    } else if (event.event_type === "workflow_completed" || event.event_type === "workflow_failed") {
        terminalEventSeen = true;
        stopFallbackPolling();
        streamAvailable = false;
        if (stream) { stream.close(); stream = null; }
        void refreshJobStatus();
    }
}
function connectEvents(token) {
    if (!jobId || token !== runToken) return;
    if (stream) stream.close();
    // Start at zero on each page load to reconstruct the full timeline.
    // EventSource handles Last-Event-ID automatically during reconnects.
    stream = new EventSource(`/api/research/jobs/${encodeURIComponent(jobId)}/events?after=0`);
    stream.addEventListener("open", () => {
        if (token !== runToken) return;
        streamAvailable = true;
        stopFallbackPolling();
        setConnection("● Live events");
    });
    for (const type of EVENT_TYPES) {
        stream.addEventListener(type, message => {
            if (token !== runToken) return;
            try { ingestEvent(JSON.parse(message.data)); }
            catch (error) { console.error("Invalid pipeline event:", error); }
        });
    }
    stream.addEventListener("error", () => {
        if (token !== runToken) return;
        if (resultShown || terminalEventSeen) {
            streamAvailable = false;
            if (stream) { stream.close(); stream = null; }
            setConnection("");
        } else {
            streamAvailable = false;
            setConnection("Reconnecting…");
            void checkStatusAfterStreamFailure(token);
        }
    });
}
async function readJson(response) {
    const type = response.headers.get("content-type") || "";
    if (!type.includes("application/json")) throw new Error("Server returned an unexpected response.");
    const data = await response.json();
    if (!response.ok) {
        const detail = data.detail || `Request failed (${response.status}).`;
        const message = typeof detail === "string" ? detail : (detail.message || JSON.stringify(detail));
        const error = new Error(message);
        error.status = response.status;
        error.guardrail = Boolean(detail?.guardrail);
        throw error;
    }
    return data;
}
async function getStatus(id) {
    const response = await fetch(`/api/research/jobs/status/${encodeURIComponent(id)}`, { cache: "no-store" });
    return readJson(response);
}
async function refreshJobStatus() {
    if (!jobId || statusLoading) return null;
    const id = jobId;
    statusLoading = true;
    try {
        const snapshot = await getStatus(id);
        if (id !== jobId) return null;
        if (snapshot.created_at) jobStartedAt = parseTimestamp(snapshot.created_at) || jobStartedAt;
        if (snapshot.question && !ui.question.value.trim()) ui.question.value = snapshot.question;
        if (TERMINAL.has(snapshot.status)) {
            renderTerminalJob(snapshot);
        } else if (events.length === 0) {
            showStatus(snapshot.status === "queued" ? "Job queued" : "Research in progress",
                snapshot.status === "queued" ? "Waiting for an available research slot." : "The agent is working through its research pipeline.", "loading");
        }
        return snapshot;
    } catch (error) {
        if (id === jobId) {
            console.error("Unable to refresh research job:", error);
            setConnection("Status temporarily unavailable");
        }
        return null;
    } finally { statusLoading = false; }
}
function stopFallbackPolling() {
    clearTimeout(pollTimer);
    pollTimer = null;
    pollDelay = FALLBACK_POLL_INITIAL_MS;
}
// Status checks are a bounded fallback only while the replayable SSE stream is down.
function scheduleFallbackPolling(token) {
    if (pollTimer || streamAvailable || resultShown || terminalEventSeen) return;
    pollTimer = setTimeout(async () => {
        pollTimer = null;
        if (token !== runToken || streamAvailable || resultShown || terminalEventSeen) return;
        await refreshJobStatus();
        if (!streamAvailable && !resultShown && !terminalEventSeen) {
            pollDelay = Math.min(pollDelay * 2, FALLBACK_POLL_MAX_MS);
            scheduleFallbackPolling(token);
        }
    }, pollDelay);
}
async function checkStatusAfterStreamFailure(token) {
    await refreshJobStatus();
    if (token === runToken && !streamAvailable && !resultShown && !terminalEventSeen) {
        scheduleFallbackPolling(token);
    }
}
function startElapsedTimer() {
    clearInterval(elapsedTimer);
    elapsedTimer = setInterval(updateElapsed, 1000);
}
function showJob(id, token) {
    jobId = id;
    setJobUrl(id);
    resetReport();
    resetPipeline();
    setBusy(true);
    setConnection("Connecting…");
    showStatus("Connecting to research", "Subscribing to the live workflow timeline.", "loading");
    startElapsedTimer();
    connectEvents(token);
}
function appendTextItem(parent, text, tag = "li") {
    parent.append(create(tag, "", text));
}
function renderMetadata(data) {
    clear(ui.metadata);
    for (const [key, value] of [
        ["Job", data.job_id], ["Session", data.research_session_id], ["Stop reason", label(data.stop_reason)],
    ]) {
        if (value !== undefined && value !== null) appendTextItem(ui.metadata, `${key}: ${value}`, "span");
    }
}
function renderFindings(findings) {
    clear(ui.findings);
    if (!Array.isArray(findings) || !findings.length) {
        ui.findings.append(create("p", "empty-message", "No findings passed the final evidence checks."));
        return;
    }
    const sections = new Map();
    findings.forEach((finding, index) => {
        const heading = finding.section?.trim() || "Detailed Analysis";
        if (!sections.has(heading)) {
            const section = create("section", "finding-section");
            section.append(renderReportText(create("h4"), heading));
            sections.set(heading, section);
            ui.findings.append(section);
        }
        const article = create("article", "finding");
        article.append(renderReportText(create("h5"), finding.title || `Finding ${index + 1}`),
            renderReportMarkdown(create("div", "report-markdown"), finding.content || ""));
        sections.get(heading).append(article);
    });
}
function renderLimitations(limitations) {
    clear(ui.limitations);
    if (!Array.isArray(limitations) || !limitations.length) {
        appendTextItem(ui.limitations, "No additional limitations were recorded.");
        return;
    }
    limitations.forEach(item => ui.limitations.append(renderReportMarkdown(create("li", "report-markdown"), item)));
}
function renderUnverified(items) {
    clear(ui.unverified);
    if (!Array.isArray(items) || !items.length) {
        hide(ui.unverifiedSection);
        return;
    }
    items.forEach((item, index) => {
        const article = create("article", "finding");
        article.append(renderReportText(create("h5"), item.title || `Unverified statement ${index + 1}`),
            renderReportMarkdown(create("div", "report-markdown"), item.content || ""));
        const links = (item.sources || []).map(source => {
            const url = safeHttpUrl(source.uri || source.url);
            const title = source.title || url || "Source";
            if (!url) return create("span", "", title);
            const anchor = create("a", "", title);
            anchor.href = url;
            anchor.target = "_blank";
            anchor.rel = "noopener noreferrer";
            return anchor;
        });
        if (links.length) {
            const line = create("p", "detail-muted", "Possible sources: ");
            links.forEach((link, linkIndex) => line.append(...(linkIndex ? ["; ", link] : [link])));
            article.append(line);
        }
        ui.unverified.append(article);
    });
    show(ui.unverifiedSection);
}
function safeHttpUrl(value) {
    try {
        const url = new URL(value);
        return ["http:", "https:"].includes(url.protocol) ? url.href : null;
    } catch { return null; }
}
function renderSources(sources) {
    clear(ui.sources);
    if (!Array.isArray(sources) || !sources.length) {
        appendTextItem(ui.sources, "No sources were included in the final report.");
        return;
    }
    sources.forEach(source => {
        const item = create("li");
        const url = safeHttpUrl(source.uri || source.url);
        const title = `${source.source_type === "document" ? "Document · " : ""}${source.title || url || "Source"}`;
        if (url) {
            const anchor = create("a", "", title);
            anchor.href = url;
            anchor.target = "_blank";
            anchor.rel = "noopener noreferrer";
            item.append(anchor);
        } else {
            item.textContent = title;
        }
        if (source.source_type === "document" && source.locations?.length) {
            const locations = [...new Set(source.locations.map(location => {
                const page = location.page_start
                    ? `Page ${location.page_start}${location.page_end && location.page_end !== location.page_start ? `–${location.page_end}` : ""}`
                    : null;
                return [page, location.section].filter(Boolean).join(" · ");
            }).filter(Boolean))];
            if (locations.length) item.append(create("small", "source-location", locations.join("; ")));
        }
        ui.sources.append(item);
    });
}
function renderErrors(errors) {
    clear(ui.errors);
    if (!Array.isArray(errors) || !errors.length) { hide(ui.errorsSection); return; }
    errors.forEach(error => appendTextItem(ui.errors,
        `${error.component ? `[${error.component}] ` : ""}${error.message || "Unknown workflow error."}`));
    ui.errorsSummary.textContent = `View workflow notes (${errors.length})`;
    show(ui.errorsSection);
}
function renderResult(data) {
    const success = data.status === "completed" && data.report;
    renderReportText(ui.reportHeading, success ? (data.report.title || "Research report") : "Research report");
    ui.resultStatus.textContent = success ? "Completed" : "Failed";
    ui.resultStatus.dataset.status = success ? "completed" : "failed";
    renderMetadata(data);
    renderReportMarkdown(ui.summary, success ? (data.report.executive_summary || "No summary was produced.")
        : "The research workflow could not produce a validated report.");
    renderFindings(success ? data.report.findings : []);
    renderLimitations(success ? data.report.limitations : []);
    renderUnverified(success ? data.report.unverified_material : []);
    renderSources(success ? data.report.sources : []);
    const lineage = document.getElementById("source-lineage");
    if (lineage) {
        const counts = [...events].reverse().find(event => event.details?.source_lineage)?.details.source_lineage || {};
        lineage.textContent = Object.entries({...counts, ...(success ? {cited: data.report.sources?.length || 0} : {})})
            .map(([key, count]) => `${label(key)}: ${count}`).join(" → ");
    }
    renderErrors(data.errors || []);
    ui.markdown.textContent = success ? (data.report.markdown || "") : "";
    show(ui.result);
}
function renderTerminalJob(job) {
    if (resultShown) return;
    resultShown = true;
    jobEndedAt = parseTimestamp(job.completed_at) || Date.now();
    stopFallbackPolling();
    clearInterval(elapsedTimer); elapsedTimer = null;
    updateElapsed();
    setBusy(false);
    const final = job.result || {
        job_id: job.job_id,
        research_session_id: job.research_session_id,
        status: "failed",
        stop_reason: null,
        report: null,
        errors: job.error_message ? [{ component: "research_service", message: job.error_message }] : [],
    };
    pipelineFailed = final.status !== "completed" || !final.report;
    renderPipeline();
    renderResult(final);
    if (final.status === "completed" && final.report) {
        showStatus("Research completed", "Your report is ready. You can review the research steps above.", "success");
    } else {
        showStatus("Research failed", job.error_message || "The workflow ended without a validated report.", "error");
    }
    // Keep the stream briefly so a fast terminal status poll cannot drop the
    // last persisted progress events. Restart-recovered jobs may have no terminal event.
    if (terminalEventSeen) {
        streamAvailable = false;
        if (stream) { stream.close(); stream = null; }
        setConnection("");
    } else {
        clearTimeout(streamCloseTimer);
        streamCloseTimer = setTimeout(() => {
            streamAvailable = false;
            if (stream) { stream.close(); stream = null; }
            setConnection("");
        }, 3500);
    }
}
async function startResearch(question, mode, documentIds) {
    runToken += 1;
    const token = runToken;
    stopNetwork();
    jobId = null;
    clearSavedJob();
    resetReport();
    setBusy(true);
    showStatus("Starting research", "Creating a research job…", "loading");
    setConnection("");
    hide(ui.pipeline);
    try {
        const headers = { "Content-Type": "application/json" };
        if (usesDocuments(mode)) headers["X-Owner-Id"] = ownerId;
        const response = await fetch("/api/research/jobs", {
            method: "POST",
            headers,
            body: JSON.stringify({
                question,
                source_selection: { mode, document_ids: documentIds },
            }),
        });
        const data = await readJson(response);
        if (token !== runToken) return;
        showJob(data.job_id, token);
        void refreshJobStatus();
    } catch (error) {
        if (token !== runToken) return;
        console.error("Research submission failed:", error);
        const message = error instanceof Error ? error.message : "Unexpected error.";
        if (error?.guardrail) {
            showStatus("Not a research request", message, "error");
        } else if (usesDocuments(mode) && error?.status === 422) {
            showDocumentIssue("Documents unavailable", message);
        } else {
            showStatus("Unable to start research", message, "error");
        }
        setBusy(false);
    }
}
async function restoreJob(id) {
    runToken += 1;
    const token = runToken;
    stopNetwork();
    try {
        const snapshot = await getStatus(id);
        if (token !== runToken) return;
        const allowed = new Set(snapshot.allowed_sources || []);
        const mode = allowed.has("documents")
            ? (allowed.has("web") ? "hybrid" : "documents")
            : "web";
        const modeInput = document.querySelector(`input[name="source-mode"][value="${mode}"]`);
        if (modeInput) modeInput.checked = true;
        updateSourceControls();
        showJob(id, token);
        jobStartedAt = parseTimestamp(snapshot.created_at) || Date.now();
        ui.question.value = snapshot.question || "";
        updateElapsed();
        if (TERMINAL.has(snapshot.status)) renderTerminalJob(snapshot);
    } catch (error) {
        console.warn("Could not restore research job:", error);
        clearSavedJob();
        showStatus("Previous job unavailable", "The saved job could not be restored. You can start a new investigation.", "neutral");
        setBusy(false);
    }
}

ui.form.addEventListener("submit", async event => {
    event.preventDefault();
    const question = ui.question.value.trim();
    if (question.length < 3) {
        showStatus("Invalid question", "Please enter at least 3 characters.", "error");
        return;
    }
    const mode = sourceMode();
    if (usesDocuments(mode)) {
        setBusy(true);
        if (ui.documentUpload.files.length) {
            showStatus("Uploading documents", "Saving the selected files before research starts.", "loading");
        } else {
            showStatus("Checking documents", "Verifying the selected files before research starts.", "loading");
        }
        if (!await uploadPendingDocuments() || !await validateSelectedDocuments()) {
            setBusy(false);
            return;
        }
    }
    const documentIds = usesDocuments(mode) ? [...selectedDocumentIds] : [];
    if (usesDocuments(mode) && !documentIds.length) {
        setBusy(false);
        const message = "Select at least one document before starting document or hybrid research.";
        showDocumentIssue("Select a document", message);
        return;
    }
    void startResearch(question, mode, documentIds);
});
document.querySelectorAll('input[name="source-mode"]').forEach(input => {
    input.addEventListener("change", updateSourceControls);
});
ui.documentUpload.addEventListener("change", () => {
    if (ui.documentUpload.files.length) clearDocumentSelectionError();
});
ui.refreshDocuments.addEventListener("click", () => void loadDocuments());
document.querySelectorAll(".example-chip").forEach(button => {
    button.addEventListener("click", () => {
        ui.question.value = button.dataset.question || "";
        ui.question.focus();
    });
});
ui.copyLink.addEventListener("click", async () => {
    if (!jobId) return;
    try {
        await navigator.clipboard.writeText(window.location.href);
        const previous = ui.copyLink.textContent;
        ui.copyLink.textContent = "Link copied ✓";
        setTimeout(() => { ui.copyLink.textContent = previous; }, 2000);
    } catch {
        showStatus("Copy unavailable", "You can copy the URL from your browser address bar.", "neutral");
    }
});
window.addEventListener("beforeunload", () => stopNetwork());

const urlJob = new URLSearchParams(window.location.search).get("job");
let savedJob = null;
try { savedJob = localStorage.getItem(STORAGE_KEY); } catch { /* Optional browser storage. */ }
updateSourceControls();
if (urlJob || savedJob) void restoreJob(urlJob || savedJob);
