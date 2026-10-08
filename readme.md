# Deep Research Agent

A learning-focused multi-agent deep research application built with:

- LangGraph
- FastAPI
- OpenAI
- Tavily
- LangSmith
- Jinja2

## Phase 1

Phase 1 implements an autonomous web research workflow:

```text
User Question
    ↓
Input Guardrail
    ↓
Research Planner
    ↓
Query Generator
    ↓
Tavily Search
    ↓
Tavily Extraction
    ↓
Evidence Extractor
    ↓
Coverage Evaluator
    │
    ├── insufficient → research again
    │
    └── sufficient
            ↓
        Writer Agent
            ↓
      Citation Checker
            ↓
       Final Report

       