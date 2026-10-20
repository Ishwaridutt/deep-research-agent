"use strict";

// Run the actual render functions with a small DOM fake, without booting the app
// or making requests. Browser checks cover the real native details interaction.
const {test} = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

class Element {
    constructor(tag) {
        this.tag = tag;
        this.children = [];
        this.dataset = {};
        this.listeners = {};
        this.isConnected = true;
        this.open = false;
        this.text = "";
    }
    set textContent(value) { this.text = String(value); this.children = []; }
    get textContent() { return [this.text, ...this.children.map(child => child.textContent)].join(" "); }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.text = ""; this.children = children; }
    addEventListener(name, callback) { this.listeners[name] = callback; }
}

const source = fs.readFileSync(path.join(__dirname, "../static/js/app.js"), "utf8");
function renderer(events) {
    const context = {
        document: {
            createElement: tag => new Element(tag),
            createTextNode: text => { const node = new Element("#text"); node.textContent = text; return node; },
            getElementById: () => null,
        }, events,
        expandedStageDetails: new Set(), pipelineFailed: false,
        STAGES: {research_planner: "Research planning", writer_synthesis: "Synthesize report"},
    };
    vm.createContext(context);
    vm.runInContext(source.slice(source.indexOf("const PLANNING"), source.indexOf("const WRITING")), context);
    for (const name of ["create", "clear", "label", "safeHttpUrl", "renderReportText", "renderReportMarkdown", "renderFindings", "renderLimitations", "renderResult", "getStageState", "appendStageMetrics", "renderStageDetails", "renderStages", "researchRounds", "researchStages", "activityDescription"]) {
        const start = source.indexOf(`function ${name}(`);
        const end = source.indexOf("\nfunction ", start + 1);
        vm.runInContext(source.slice(start, end), context);
    }
    return context;
}
function event(stage, details, iteration = 1) {
    return {stage, details, iteration, event_type: "stage_completed"};
}

test("planner output is text-only and preserves user expansion across updates", () => {
    const context = renderer([event("research_planner", {
        active_subquestion_count: 1,
        subquestions: [{question: "<img src=x onerror=alert(1)>", objective: "Assess outcomes", parent_question: "Original scope"}],
    })]);
    const output = context.renderStageDetails("research_planner", null);
    assert.match(output.textContent, /Assess outcomes/);
    assert.match(output.textContent, /Follow-up to: Original scope/);
    assert.match(output.textContent, /<img src=x onerror=alert\(1\)>/);
    assert.equal(output.open, false);
    output.open = true;
    output.listeners.toggle();
    const updated = context.renderStageDetails("research_planner", null);
    assert.equal(updated.open, true);
    output.isConnected = false;
    output.open = false;
    output.listeners.toggle();
    assert.equal(context.renderStageDetails("research_planner", null).open, true);
});

test("scoring audit groups scores, decisions, and public previews under the task", () => {
    const context = renderer([event("research_evidence_extractor", {
        accepted_chunk_count: 1, min_evidence_relevance: .6,
        scoring_groups: [{question: "What works?", passages: [
            {chunk_id: "chunk-1", title: "Study", relevance_score: .9, decision: "accepted", source_type: "web",
                preview: "<script>not executable</script>", preview_truncated: true, queries: ["outcomes trial"]},
            {title: "Private document", relevance_score: .2, decision: "below_threshold", source_type: "document"},
        ]}],
    })]);
    const output = context.renderStageDetails("research_evidence_extractor", 1);
    for (const phrase of ["What works?", "Accepted · 0.90", "Rejected (below threshold) · 0.20", "chunk-1",
        "outcomes trial", "<script>not executable</script>", "scoring used the full passage", "Private passage text"]) {
        assert.ok(output.textContent.includes(phrase), phrase);
    }
});

test("report GFM tables pass through the sanitizer and preserve alignment and inline markup", () => {
    const context = renderer([]);
    context.marked = require("../static/js/vendor/marked-18.1.1.js").marked;
    let html, options;
    context.DOMPurify = {sanitize: (value, config) => {
        html = value; options = config;
        return new Element("fragment");
    }};
    const parent = new Element("div");
    parent.querySelectorAll = () => [];
    context.renderReportMarkdown(parent, "| Body weight | Daily protein at 1.6 g/kg | Protein per meal across four equal meals |\n|---|---:|---:|\n| 60 kg | **96 g** | 24 g |\n| 70 kg | 112 g | 28 g |\n| 80 kg | 128 g | 32 g |\n| 90 kg | 144 g | 36 g |");
    assert.match(html, /<table>/);
    assert.equal((html.match(/<tr>/g) || []).length, 5);
    assert.match(html, /align="right"/);
    assert.match(html, /<strong>96 g<\/strong>/);
    assert.equal(options.RETURN_DOM_FRAGMENT, true);
    assert.equal(options.ALLOWED_TAGS.includes("script"), false);
    assert.equal(options.ALLOWED_TAGS.includes("img"), false);
    assert.equal(options.ALLOWED_ATTR.includes("onerror"), false);
    assert.equal(options.ALLOWED_URI_REGEXP.test("javascript:alert(1)"), false);
});

test("queries stay grouped by subquestion and scoped to their research round", () => {
    const context = renderer([
        event("research_query_generator", {query_count: 1, query_groups: [{question: "Old task", queries: [{query: "old query"}]}]}, 1),
        event("research_query_generator", {query_count: 2, query_groups: [
            {question: "Mechanisms?", queries: [{query: "mechanism study", purpose: "Find a mechanism"}]},
            {question: "Outcomes?", queries: [{query: "outcome trial", purpose: "Measure outcomes"}]},
        ]}, 2),
    ]);
    const output = context.renderStageDetails("research_query_generator", 2);
    assert.match(output.textContent, /Mechanisms\?/);
    assert.match(output.textContent, /mechanism study/);
    assert.match(output.textContent, /Outcomes\?/);
    assert.match(output.textContent, /Measure outcomes/);
    assert.doesNotMatch(output.textContent, /old query/);
    assert.equal(output.children[1].children.length, 2);
});

test("search provider totals are labelled retained matches, with failures and fallback", () => {
    const context = renderer([event("research_web_search", {
        result_count: 17, provider_counts: {tavily: 10, wikipedia: 3, arxiv: 4},
        failed_branches: ["web"], web_fallback_activated: true,
    })]);
    const text = context.renderStageDetails("research_web_search", 1).textContent;
    for (const word of ["Tavily", "Wikipedia", "arXiv", "17 retained", "after filtering", "Brave fallback", "web"]) {
        assert.ok(text.includes(word), word);
    }
});

test("evidence progress separates filtering, capacity skips, invalid decisions, and failures", () => {
    const context = renderer([event("research_evidence_extractor", {
        scored_chunk_count: 10, accepted_chunk_count: 6, low_relevance_chunk_count: 3,
        capacity_skipped_chunk_count: 1, total_evidence_count: 14, min_evidence_relevance: 0.6,
        missing_chunk_count: 2, unknown_chunk_count: 1, duplicate_chunk_count: 1,
        failed_batch_count: 1, capacity_reached: true,
    })]);
    const text = context.renderStageDetails("research_evidence_extractor", 1).textContent;
    for (const word of ["6 passages kept", "Below threshold", "Capacity skipped", "0.60", "not necessarily factually incorrect", "14", "2 missing", "1 scoring batches failed", "capacity was reached"]) {
        assert.ok(text.includes(word), word);
    }
});

test("coverage shows individual insufficient objectives and nonblocking limitations", () => {
    const data = {
        coverage_status: "sufficient", assessment_counts: {supported: 2, partial: 1, missing: 0, conflicting: 0},
        assessments: [{question: "What is uncertain?", status: "partial", evidence_strength: "weak", supporting_evidence_count: 1, reason: "Long-term outcomes are unavailable."}],
        gaps: [{question: "What is uncertain?", description: "Long-term measurement missing.", priority: 2}],
    };
    const context = renderer([event("research_coverage_evaluator", data)]);
    let text = context.renderStageDetails("research_coverage_evaluator", 1).textContent;
    for (const word of ["Ready to write", "Partial", "What is uncertain?", "Long-term outcomes are unavailable.", "Remaining limitations", "Priority 2/5"]) {
        assert.ok(text.includes(word), word);
    }
    data.coverage_status = "insufficient";
    text = context.renderStageDetails("research_coverage_evaluator", 1).textContent;
    assert.match(text, /Material gaps/);
    assert.doesNotMatch(text, /Remaining limitations/);
});

test("old events and writer stages remain usable without detail payloads", () => {
    const context = renderer([
        event("research_evidence_extractor", {}), event("writer_synthesis", {}),
        event("research_planner", {active_subquestion_count: 3}),
    ]);
    assert.equal(context.renderStageDetails("research_evidence_extractor", 1), null);
    assert.equal(context.renderStageDetails("writer_synthesis", null), null);
    assert.match(context.renderStageDetails("research_planner", null).textContent, /Only counts were recorded/);
    const stages = context.renderStages(["writer_synthesis"], null);
    assert.equal(stages.children[0].children.length, 1);
    assert.equal(stages.children[0].dataset.state, "completed");
});

test("recent activity summarizes research outputs without expanding writer details", () => {
    const context = renderer([]);
    assert.match(context.activityDescription(event("research_query_generator", {query_count: 4}))[1], /4 queries grouped by subquestion/);
    assert.match(context.activityDescription(event("research_evidence_extractor", {accepted_chunk_count: 6, low_relevance_chunk_count: 3}))[1], /6 kept · 3 below threshold/);
    assert.match(context.activityDescription(event("research_coverage_evaluator", {assessment_counts: {supported: 2}, gap_count: 1}))[1], /2 supported · 1 gaps \/ limitations/);
    assert.equal(context.activityDescription(event("writer_synthesis", {}))[0], "Synthesize report completed");
});

test("initial planning belongs to round one, without duplicating follow-up planning", () => {
    const context = renderer([event("research_planner", {}, 1), event("research_planner", {}, 2)]);
    assert.equal(context.researchRounds().join(","), "1,2");
    for (const round of [1, 2]) {
        assert.equal(context.researchStages(round)[0], "research_planner");
        assert.equal(context.researchStages(round).filter(stage => stage === "research_planner").length, 1);
    }
    assert.equal(vm.runInContext("PLANNING.includes('research_planner')", context), false);
});

test("private document row is skipped without attachments and shares real retrieval lifecycle", () => {
    const context = renderer([event("research_source_selection", {source_mode: "web", document_count: 0})]);
    assert.equal(context.getStageState("research_document_retrieval", 1), "skipped");
    const row = context.renderStages(["research_document_retrieval"], 1).children[0];
    assert.equal(row.dataset.state, "skipped");
    assert.match(row.textContent, /↷.*Skipped.*No private documents attached/);
    context.events.push(event("research_source_selection", {source_mode: "documents", document_count: 1}, 2));
    assert.equal(context.researchStages(2).includes("research_retrieval"), false);
    assert.equal(context.researchStages(2).includes("research_document_retrieval"), true);
    context.events.push({stage: "research_retrieval", iteration: 2, event_type: "stage_started"});
    assert.equal(context.getStageState("research_document_retrieval", 2), "running");
    context.pipelineFailed = true;
    assert.equal(context.getStageState("research_document_retrieval", 2), "failed");
    assert.equal(context.getStageState("research_document_retrieval", 1), "skipped");
});

test("retrieval shows separate branch query counts and explains overlap", () => {
    const context = renderer([event("research_retrieval", {
        web_chunk_count: 4, document_chunk_count: 2,
        web_groups: [{question: "Public task?", passage_count: 4, estimated_tokens: 250,
            query_matches: [{query: "public query", retrieved_count: 8, selected_count: 4}]}],
        document_groups: [{question: "Document task?", passage_count: 2, estimated_tokens: 120,
            query_matches: [{query: "document query", retrieved_count: 5, selected_count: 2}]}],
    })]);
    let text = context.renderStageDetails("research_retrieval", 1).textContent;
    assert.match(text, /4 passages selected/);
    assert.match(text, /8 retrieved · 4 included after context limits/);
    assert.match(text, /counts overlap and must not be added/);
    assert.doesNotMatch(text, /document query/);
    text = context.renderStageDetails("research_document_retrieval", 1).textContent;
    assert.match(text, /Document task\?/);
    assert.match(text, /5 retrieved · 2 included/);
    assert.doesNotMatch(text, /public query/);
});

test("search distinguishes discovered providers from retained results and zero matches", () => {
    const context = renderer([event("research_web_search", {
        result_count: 5, provider_counts: {tavily: 5}, discovered_count: 8,
        discovered_provider_counts: {tavily: 5, arxiv: 3, wikipedia: 0},
    })]);
    const text = context.renderStageDetails("research_web_search", 1).textContent;
    for (const phrase of ["8 discovered", "arXiv", "Wikipedia", "Zero means no results", "Retained by provider: Tavily 5"]) {
        assert.ok(text.includes(phrase), phrase);
    }
});

test("extraction shows totals and individual providers without counting reused pages as new", () => {
    const context = renderer([event("research_source_processor", {
        source_count: 7, extracted_count: 5, extracted_provider_counts: {tavily: 2, arxiv: 2, wikipedia: 1},
        reused_count: 2, attempted_count: 6, deferred_count: 10,
    })]);
    const text = context.renderStageDetails("research_source_processor", 1).textContent;
    for (const phrase of ["5 pages extracted", "Tavily", "Wikipedia", "arXiv", "7 unique pages", "Reused pages", "Budget deferred"]) {
        assert.ok(text.includes(phrase), phrase);
    }
});

test("provider counts identify their job and round instead of implying cumulative totals", () => {
    const sample = event("research_web_search", {result_count: 42, discovered_count: 108,
        discovered_provider_counts: {tavily: 60, arxiv: 36, wikipedia: 12}, executed_query_count: 12});
    sample.job_id = "new-job";
    const context = renderer([sample]);
    const text = context.renderStageDetails("research_web_search", 1).textContent;
    assert.match(text, /Job new-job · Round 1 only · 12 queries executed/);
    assert.match(text, /Not a cumulative job total/);
    assert.doesNotMatch(text, /old-job/);
});

test("report emphasis creates safe DOM elements while preserving citations and HTML as text", () => {
    const context = renderer([]);
    const node = context.renderReportText(new Element("p"), "**Practical approach:** Use *careful* wording and `**literal code**`. [1][2] <img src=x onerror=alert(1)>");
    assert.equal(node.children.find(child => child.tag === "strong").textContent.trim(), "Practical approach:");
    assert.equal(node.children.find(child => child.tag === "em").textContent.trim(), "careful");
    assert.equal(node.children.find(child => child.tag === "code").textContent, "**literal code**");
    assert.match(node.textContent, /\[1\]\[2\]/);
    assert.match(node.textContent, /<img src=x onerror=alert\(1\)>/);
    assert.equal(node.children.some(child => child.tag === "img"), false);
    const escaped = context.renderReportText(new Element("p"), String.raw`\*\*not bold\*\* and an unmatched **`);
    assert.equal(escaped.children.some(child => child.tag === "strong"), false);
    context.renderReportText(node, "New text");
    assert.equal(node.children.some(child => child.tag === "strong"), false);
});

test("findings, summary and limitations render emphasis but the raw Markdown view stays raw", () => {
    const context = renderer([]);
    context.ui = Object.fromEntries(["findings", "limitations", "reportHeading", "resultStatus", "summary", "markdown", "result"].map(key => [key, new Element("div")]));
    context.renderMetadata = context.renderSources = context.renderErrors = context.show = () => {};
    const markdown = "# Report\n\n**Practical approach:** Use evidence. [1]";
    context.renderResult({status: "completed", report: {title: "Report", executive_summary: "**Summary:** Grounded.",
        findings: [{section: "Analysis", title: "Finding", content: "**Practical approach:** Use evidence. [1]"}],
        limitations: ["**Limitation:** Missing data."], markdown}});
    const summary = context.ui.summary;
    assert.equal(summary.children.find(child => child.tag === "strong").textContent.trim(), "Summary:");
    const content = context.ui.findings.children[0].children[1].children[1];
    assert.equal(content.children.find(child => child.tag === "strong").textContent.trim(), "Practical approach:");
    assert.equal(context.ui.limitations.children[0].children.find(child => child.tag === "strong").textContent.trim(), "Limitation:");
    assert.equal(context.ui.markdown.textContent, markdown);
});
