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
    copyLink: $("copy-job-link"),
    coverageStatus: $("coverage-status"),
    coverageDescription: $("coverage-description"),
    coverageGapCount: $("coverage-gap-count"),
    coverageAssessmentCount: $("coverage-assessment-count"),
    activity: $("activity-feed"),
    activityEmpty: $("activity-empty"),
    eventCount: $("event-count"),
    result: $("result-panel"),
    resultStatus: $("result-status"),
    metadata: $("job-metadata"),
    summary: $("report-summary"),
    findings: $("report-findings"),
    limitations: $("report-limitations"),
    sources: $("report-sources"),
    errorsSection: $("workflow-errors-section"),
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
    research_source_processor: "Prepare public sources",
    research_retrieval: "Find relevant passages",
    research_evidence_extractor: "Extract evidence",
    research_coverage_evaluator: "Evaluate coverage",
    research_routing: "Decide next step",
    writer_prepare_evidence: "Prepare evidence",
    writer_synthesis: "Synthesize report",
    writer_citation_checker: "Verify citations",
    finalize_report: "Finalize report",
};

const PLANNING = ["input_guardrail", "research_planner"];
const RESEARCH = [
    "research_query_generator", "research_source_selection", "research_web_search", "research_source_processor",
    "research_retrieval", "research_evidence_extractor", "research_coverage_evaluator",
    "research_routing",
];
const WRITING = ["writer_prepare_evidence", "writer_synthesis", "writer_citation_checker", "finalize_report"];
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
let streamAvailable = false;
let pollDelay = FALLBACK_POLL_INITIAL_MS;
let collapsedRounds = new Set();
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
    } catch (error) {
        setDocumentStatus(error instanceof Error ? error.message : "Unable to load documents.", "error");
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
        await loadDocuments();
        return true;
    } catch (error) {
        const message = error instanceof Error ? error.message : "Document upload failed.";
        setDocumentStatus(message, "error");
        showStatus("Document upload failed", message, "error");
        return false;
    }
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
    hide(ui.errorsSection);
    ui.markdownSection.open = false;
    ui.summary.textContent = "";
    ui.markdown.textContent = "";
    [ui.findings, ui.limitations, ui.sources, ui.errors, ui.metadata].forEach(clear);
}
function resetPipeline() {
    events = [];
    highestSequence = 0;
    resultShown = false;
    terminalEventSeen = false;
    collapsedRounds = new Set();
    previousLatestRound = null;
    jobStartedAt = Date.now();
    jobEndedAt = null;
    ui.coverageStatus.textContent = "Awaiting evaluation";
    ui.coverageDescription.textContent = "The evidence evaluator will check research coverage after extraction.";
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
    let state = "pending";
    for (const event of events) {
        if (event.stage !== stage) continue;
        if (round !== null && event.iteration !== round) continue;
        if (event.event_type === "stage_started") state = "running";
        if (event.event_type === "stage_completed") state = "completed";
        if (event.event_type === "stage_failed") state = "failed";
    }
    return state;
}
function renderStages(stages, round = null) {
    const list = create("ol", "stage-list");
    for (const stage of stages) {
        const state = getStageState(stage, round);
        const row = create("li", "stage-row");
        row.dataset.state = state;
        row.append(
            create("span", "stage-node", ({ completed: "✓", failed: "!", running: "◆", pending: "·" })[state]),
            create("span", "stage-name", STAGES[stage]),
            create("span", "stage-caption", ({ completed: "Done", failed: "Failed", running: "Running", pending: "Waiting" })[state]),
        );
        list.append(row);
    }
    return list;
}
function renderGroup(title, hint, stages) {
    const group = create("section", "pipeline-group");
    const head = create("div", "group-head");
    head.append(create("h3", "", title), create("span", "group-hint", hint));
    group.append(head, renderStages(stages));
    return group;
}
function researchRounds() {
    return [...new Set(events.filter(event => RESEARCH.includes(event.stage) && Number.isInteger(event.iteration))
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
    const selection = events.find(event =>
        event.iteration === round
        && event.stage === "research_source_selection"
        && event.event_type === "stage_completed");
    if (selection?.details?.source_mode === "documents") {
        return RESEARCH.filter(stage => !["research_web_search", "research_source_processor"].includes(stage));
    }
    return RESEARCH;
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
    summary.append(left, create("span", "round-status", status));
    details.append(summary);
    const body = create("div", "round-details");
    body.append(renderStages(researchStages(round), round));
    const decision = retry || stop;
    if (decision) {
        const note = create("div", "round-note");
        note.dataset.kind = retry ? "retry" : "done";
        note.append(create("strong", "", retry ? `↻ Continuing to round ${decision.details?.next_iteration || round + 1}` : "Research pass concluded"));
        const gaps = decision.details?.gap_count;
        const reason = decision.details?.reason || (retry ? "Further evidence is needed." : "Research stage finished.");
        note.append(create("p", "", `${reason}${typeof gaps === "number" ? ` · ${gaps} remaining gap${gaps === 1 ? "" : "s"}` : ""}`));
        body.append(note);
    } else if (coverage?.details?.coverage_status) {
        const note = create("div", "round-note");
        const gapCount = coverage.details.gap_count;
        note.append(create("strong", "", `Coverage: ${label(coverage.details.coverage_status)}`));
        if (Number.isFinite(gapCount)) note.append(create("p", "", `${gapCount} unresolved gap${gapCount === 1 ? "" : "s"}`));
        body.append(note);
    }
    details.append(body);
    return details;
}
function renderPipeline() {
    clear(ui.timeline);
    ui.timeline.append(renderGroup("01 · Plan", "Understand the request", PLANNING));

    const research = create("section", "pipeline-group");
    const head = create("div", "group-head");
    head.append(create("h3", "", "02 · Research & evaluate"), create("span", "group-hint", "Evidence-driven iterations"));
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
        research.append(create("p", "empty-rounds", "Research rounds will appear after planning."));
    }
    ui.timeline.append(research, renderGroup("03 · Write & verify", "Grounded report", WRITING));
    renderInsights();
    renderActivity();
}
function renderInsights() {
    const coverage = [...events].reverse().find(event =>
        event.event_type === "stage_completed" && event.stage === "research_coverage_evaluator"
        && event.details?.coverage_status);
    if (!coverage) return;
    const status = coverage.details.coverage_status;
    ui.coverageStatus.textContent = status === "sufficient" ? "Coverage sufficient"
        : status === "insufficient" ? "More evidence needed" : label(status);
    const gaps = coverage.details.gap_count;
    const assessments = coverage.details.assessment_count;
    ui.coverageGapCount.textContent = Number.isInteger(gaps) ? gaps : "—";
    ui.coverageAssessmentCount.textContent = Number.isInteger(assessments) ? assessments : "—";
    ui.coverageDescription.textContent = Number.isInteger(gaps)
        ? `Round ${coverage.iteration}: ${gaps} unresolved research gap${gaps === 1 ? "" : "s"} identified.`
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
    if (event.event_type === "stage_completed" && event.stage === "research_source_selection") {
        const count = event.details?.document_count || 0;
        return ["Sources prepared", count ? `${count} private document${count === 1 ? "" : "s"} ready` : "Web sources selected"];
    }
    if (event.event_type === "stage_completed" && event.stage === "research_web_search") {
        const count = event.details?.result_count || 0;
        const providers = Object.entries(event.details?.provider_counts || {})
            .map(([provider, total]) => `${label(provider)} ${total}`).join(" · ");
        const notes = [`${count} result${count === 1 ? "" : "s"}`, providers];
        if (event.details?.web_fallback_activated) notes.push("Brave fallback used");
        const failures = event.details?.partial_failure_count || 0;
        if (failures) notes.push(`${failures} provider branch${failures === 1 ? "" : "es"} unavailable`);
        return ["Public search completed", notes.filter(Boolean).join(" · ")];
    }
    if (event.event_type === "stage_completed" && event.stage === "research_source_processor") {
        const count = event.details?.source_count || 0;
        const fallback = event.details?.extraction_fallback_count || 0;
        return ["Public sources prepared", `${count} source${count === 1 ? "" : "s"}${fallback ? ` · Firecrawl fallback ${fallback}` : ""}`];
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
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
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
    findings.forEach((finding, index) => {
        const article = create("article", "finding");
        article.append(create("h4", "", finding.title || `Finding ${index + 1}`), create("p", "", finding.content || ""));
        ui.findings.append(article);
    });
}
function renderLimitations(limitations) {
    clear(ui.limitations);
    if (!Array.isArray(limitations) || !limitations.length) {
        appendTextItem(ui.limitations, "No additional limitations were recorded.");
        return;
    }
    limitations.forEach(item => appendTextItem(ui.limitations, item));
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
    show(ui.errorsSection);
}
function renderResult(data) {
    const success = data.status === "completed" && data.report;
    ui.resultStatus.textContent = success ? "Completed" : "Failed";
    ui.resultStatus.dataset.status = success ? "completed" : "failed";
    renderMetadata(data);
    ui.summary.textContent = success ? (data.report.executive_summary || "No summary was produced.")
        : "The research workflow could not produce a validated report.";
    renderFindings(success ? data.report.findings : []);
    renderLimitations(success ? data.report.limitations : []);
    renderSources(success ? data.report.sources : []);
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
        showStatus("Unable to start research", error instanceof Error ? error.message : "Unexpected error.", "error");
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
    if (usesDocuments(mode) && ui.documentUpload.files.length) {
        setBusy(true);
        showStatus("Uploading documents", "Saving the selected files before research starts.", "loading");
        if (!await uploadPendingDocuments()) {
            setBusy(false);
            return;
        }
    }
    const documentIds = usesDocuments(mode) ? [...selectedDocumentIds] : [];
    if (usesDocuments(mode) && !documentIds.length) {
        setBusy(false);
        const message = "Select at least one document before starting document or hybrid research.";
        setDocumentStatus(message, "error", DOCUMENT_SELECTION_ISSUE);
        showStatus("Select a document", message, "error", DOCUMENT_SELECTION_ISSUE);
        replayAttention(ui.documentPanel);
        replayAttention(ui.status);
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
