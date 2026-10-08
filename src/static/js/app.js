"use strict";


const form = document.getElementById(
    "research-form"
);

const questionInput = document.getElementById(
    "research-question"
);

const researchButton = document.getElementById(
    "research-button"
);

const statusPanel = document.getElementById(
    "status-panel"
);

const statusSpinner = document.getElementById(
    "status-spinner"
);

const statusTitle = document.getElementById(
    "status-title"
);

const statusMessage = document.getElementById(
    "status-message"
);

const resultPanel = document.getElementById(
    "result-panel"
);

const resultStatus = document.getElementById(
    "result-status"
);

const jobMetadata = document.getElementById(
    "job-metadata"
);

const reportSummary = document.getElementById(
    "report-summary"
);

const reportFindings = document.getElementById(
    "report-findings"
);

const reportLimitations = document.getElementById(
    "report-limitations"
);

const reportSources = document.getElementById(
    "report-sources"
);

const workflowErrorsSection = document.getElementById(
    "workflow-errors-section"
);

const workflowErrors = document.getElementById(
    "workflow-errors"
);

const markdownSection = document.getElementById(
    "markdown-section"
);

const reportMarkdown = document.getElementById(
    "report-markdown"
);


function showElement(element) {
    element.classList.remove(
        "hidden"
    );
}


function hideElement(element) {
    element.classList.add(
        "hidden"
    );
}


function clearElement(element) {
    element.replaceChildren();
}


function formatValue(value) {
    if (!value) {
        return "Not available";
    }

    return String(value)
        .replaceAll("_", " ")
        .replace(
            /\b\w/g,
            character => character.toUpperCase()
        );
}


function setLoadingState(isLoading) {
    researchButton.disabled = isLoading;
    questionInput.disabled = isLoading;

    if (isLoading) {
        researchButton.textContent = (
            "Researching..."
        );

        showElement(
            statusSpinner
        );
    } else {
        researchButton.textContent = (
            "Start Research"
        );

        hideElement(
            statusSpinner
        );
    }
}


function showStatus(
    title,
    message,
    type = "neutral"
) {
    statusTitle.textContent = title;
    statusMessage.textContent = message;

    statusPanel.dataset.type = type;

    showElement(
        statusPanel
    );
}


function resetResults() {
    hideElement(
        resultPanel
    );

    hideElement(
        workflowErrorsSection
    );

    markdownSection.open = false;

    reportSummary.textContent = "";
    reportMarkdown.textContent = "";

    clearElement(
        reportFindings
    );

    clearElement(
        reportLimitations
    );

    clearElement(
        reportSources
    );

    clearElement(
        workflowErrors
    );

    clearElement(
        jobMetadata
    );
}


function renderMetadata(data) {
    const jobItem = document.createElement(
        "span"
    );

    jobItem.textContent = (
        `Job: ${data.job_id}`
    );

    const sessionItem = document.createElement(
        "span"
    );

    sessionItem.textContent = (
        `Session: ${data.research_session_id}`
    );

    const stopReasonItem = document.createElement(
        "span"
    );

    stopReasonItem.textContent = (
        `Stop reason: ${
            formatValue(
                data.stop_reason
            )
        }`
    );

    jobMetadata.append(
        jobItem,
        sessionItem,
        stopReasonItem
    );
}


function renderFindings(findings) {
    clearElement(
        reportFindings
    );

    if (
        !Array.isArray(findings)
        || findings.length === 0
    ) {
        const emptyMessage = (
            document.createElement("p")
        );

        emptyMessage.className = (
            "empty-message"
        );

        emptyMessage.textContent = (
            "No findings passed the final "
            + "evidence checks."
        );

        reportFindings.appendChild(
            emptyMessage
        );

        return;
    }

    findings.forEach(
        (finding, index) => {
            const article = (
                document.createElement(
                    "article"
                )
            );

            article.className = (
                "finding"
            );

            const heading = (
                document.createElement(
                    "h4"
                )
            );

            heading.textContent = (
                finding.title
                || `Finding ${index + 1}`
            );

            const content = (
                document.createElement(
                    "p"
                )
            );

            content.textContent = (
                finding.content || ""
            );

            article.append(
                heading,
                content
            );

            reportFindings.appendChild(
                article
            );
        }
    );
}


function renderLimitations(limitations) {
    clearElement(
        reportLimitations
    );

    if (
        !Array.isArray(limitations)
        || limitations.length === 0
    ) {
        const item = (
            document.createElement("li")
        );

        item.textContent = (
            "No additional limitations were recorded."
        );

        reportLimitations.appendChild(
            item
        );

        return;
    }

    limitations.forEach(
        limitation => {
            const item = (
                document.createElement("li")
            );

            item.textContent = (
                limitation
            );

            reportLimitations.appendChild(
                item
            );
        }
    );
}


function renderSources(sources) {
    clearElement(
        reportSources
    );

    if (
        !Array.isArray(sources)
        || sources.length === 0
    ) {
        const item = (
            document.createElement("li")
        );

        item.textContent = (
            "No sources were included "
            + "in the final report."
        );

        reportSources.appendChild(
            item
        );

        return;
    }

    sources.forEach(
        source => {
            const item = (
                document.createElement("li")
            );

            const link = (
                document.createElement("a")
            );

            link.textContent = (
                source.title
                || source.uri
            );

            link.href = source.uri;

            link.target = "_blank";

            link.rel = (
                "noopener noreferrer"
            );

            item.appendChild(
                link
            );

            reportSources.appendChild(
                item
            );
        }
    );
}


function renderWorkflowErrors(errors) {
    clearElement(
        workflowErrors
    );

    if (
        !Array.isArray(errors)
        || errors.length === 0
    ) {
        hideElement(
            workflowErrorsSection
        );

        return;
    }

    errors.forEach(
        error => {
            const item = (
                document.createElement("li")
            );

            const component = (
                error.component
                ? `[${error.component}] `
                : ""
            );

            item.textContent = (
                component
                + (
                    error.message
                    || "Unknown workflow error."
                )
            );

            workflowErrors.appendChild(
                item
            );
        }
    );

    showElement(
        workflowErrorsSection
    );
}


function renderSuccessfulResult(data) {
    const report = data.report;

    if (!report) {
        throw new Error(
            "Research completed without a report."
        );
    }

    resultStatus.textContent = (
        formatValue(data.status)
    );

    resultStatus.dataset.status = (
        data.status
    );

    renderMetadata(
        data
    );

    reportSummary.textContent = (
        report.executive_summary
        || "No summary was produced."
    );

    renderFindings(
        report.findings
    );

    renderLimitations(
        report.limitations
    );

    renderSources(
        report.sources
    );

    renderWorkflowErrors(
        data.errors
    );

    reportMarkdown.textContent = (
        report.markdown || ""
    );

    showElement(
        resultPanel
    );
}


function renderFailedResult(data) {
    resultStatus.textContent = (
        "Failed"
    );

    resultStatus.dataset.status = (
        "failed"
    );

    renderMetadata(
        data
    );

    reportSummary.textContent = (
        "The research workflow could not "
        + "produce a validated report."
    );

    renderFindings(
        []
    );

    renderLimitations(
        []
    );

    renderSources(
        []
    );

    renderWorkflowErrors(
        data.errors
    );

    reportMarkdown.textContent = "";

    showElement(
        resultPanel
    );
}


async function parseResponse(response) {
    const contentType = (
        response.headers.get(
            "content-type"
        ) || ""
    );

    if (
        contentType.includes(
            "application/json"
        )
    ) {
        return await response.json();
    }

    const text = await response.text();

    throw new Error(
        text
        || (
            "Server returned an unexpected response."
        )
    );
}


async function runResearch(question) {
    const response = await fetch(
        "/api/research",
        {
            method: "POST",

            headers: {
                "Content-Type": (
                    "application/json"
                )
            },

            body: JSON.stringify(
                {
                    question: question,
                    sources: [
                        "web"
                    ]
                }
            )
        }
    );

    const data = await parseResponse(
        response
    );

    if (!response.ok) {
        const detail = (
            data.detail
            || "Research request failed."
        );

        throw new Error(
            typeof detail === "string"
                ? detail
                : JSON.stringify(detail)
        );
    }

    return data;
}


form.addEventListener(
    "submit",
    async event => {
        event.preventDefault();

        const question = (
            questionInput.value.trim()
        );

        if (question.length < 3) {
            showStatus(
                "Invalid question",
                (
                    "Please enter a research "
                    + "question with at least "
                    + "3 characters."
                ),
                "error"
            );

            return;
        }

        resetResults();

        setLoadingState(
            true
        );

        showStatus(
            "Research in progress",
            (
                "The research agent is planning, "
                + "searching, evaluating evidence, "
                + "and preparing the report."
            ),
            "loading"
        );

        try {
            const result = (
                await runResearch(
                    question
                )
            );

            if (
                result.status
                === "completed"
            ) {
                showStatus(
                    "Research completed",
                    (
                        "The final report passed "
                        + "the workflow validation steps."
                    ),
                    "success"
                );

                renderSuccessfulResult(
                    result
                );
            } else {
                showStatus(
                    "Research failed",
                    (
                        "The workflow finished without "
                        + "a validated report."
                    ),
                    "error"
                );

                renderFailedResult(
                    result
                );
            }

        } catch (error) {
            console.error(
                error
            );

            showStatus(
                "Request failed",
                (
                    error instanceof Error
                        ? error.message
                        : "An unexpected error occurred."
                ),
                "error"
            );

        } finally {
            setLoadingState(
                false
            );
        }
    }
);
