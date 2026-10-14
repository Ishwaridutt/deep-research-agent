"""LangGraph construction for the Deep Research workflow."""

from functools import partial

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.research import (
    research_coverage_evaluator_node,
    research_evidence_extractor_node,
    research_planner_node,
    research_query_generator_node,
    research_source_selection_node,
    research_source_processor_node,
    research_web_search_node,
    research_retrieval_node,
)
from src.agents.writer import (
    writer_citation_checker_node,
    writer_prepare_evidence_node,
    writer_synthesis_node,
)
from src.config import Settings
from src.rendering.finalize_report import finalize_report_node
from src.guardrails.input_guardrail import input_guardrail_node
from src.graph.graph_routing import (
    research_routing_node,
    route_after_research,
    route_after_source_selection,
    route_after_terminal_check,
)
from src.graph.graph_state import WorkflowState
from src.llm import LLMService
from src.retrieval import RetrievalSessionFactory
from src.services.document_service import DocumentService
from src.tools.cleaning import SourceCleaner
from src.tools.extraction import SourceContentExtractor
from src.tools.search import SearchProvider


INPUT_GUARDRAIL = "input_guardrail"
RESEARCH_PLANNER = "research_planner"
RESEARCH_QUERY_GENERATOR = "research_query_generator"
RESEARCH_SOURCE_SELECTION = "research_source_selection"
RESEARCH_WEB_SEARCH = "research_web_search"
RESEARCH_SOURCE_PROCESSOR = "research_source_processor"
RESEARCH_RETRIEVAL = "research_retrieval"
RESEARCH_EVIDENCE_EXTRACTOR = "research_evidence_extractor"
RESEARCH_COVERAGE_EVALUATOR = "research_coverage_evaluator"
RESEARCH_ROUTING = "research_routing"
WRITER_PREPARE_EVIDENCE = "writer_prepare_evidence"
WRITER_SYNTHESIS = "writer_synthesis"
WRITER_CITATION_CHECKER = "writer_citation_checker"
FINALIZE_REPORT = "finalize_report"


def build_research_graph(
    *,
    settings: Settings,
    llm_service: LLMService,
    search_provider: SearchProvider,
    source_extractor: SourceContentExtractor,
    source_cleaner: SourceCleaner,
    retrieval_session_factory: RetrievalSessionFactory,
    document_service: DocumentService,
) -> CompiledStateGraph:
    """
    Build and compile the research workflow with selected source branches.

    External dependencies are created outside the graph and injected here.
    Agent nodes therefore remain independent of provider construction.
    """

    builder = StateGraph(WorkflowState)

    #
    # Nodes
    #

    builder.add_node(
        INPUT_GUARDRAIL,
        partial(
            input_guardrail_node,
            settings=settings,
        ),
    )

    builder.add_node(
        RESEARCH_PLANNER,
        partial(
            research_planner_node,
            llm_service=llm_service,
            settings=settings,
        ),
    )

    builder.add_node(
        RESEARCH_QUERY_GENERATOR,
        partial(
            research_query_generator_node,
            llm_service=llm_service,
            settings=settings,
        ),
    )

    builder.add_node(
        RESEARCH_WEB_SEARCH,
        partial(
            research_web_search_node,
            search_provider=search_provider,
            settings=settings,
        ),
    )

    builder.add_node(
        RESEARCH_SOURCE_SELECTION,
        partial(
            research_source_selection_node,
            document_service=document_service,
            retrieval_session_factory=retrieval_session_factory,
        ),
    )

    builder.add_node(
        RESEARCH_SOURCE_PROCESSOR,
        partial(
            research_source_processor_node,
            source_extractor=source_extractor,
            source_cleaner=source_cleaner,
        ),
    )

    builder.add_node(
        RESEARCH_RETRIEVAL,
        partial(
            research_retrieval_node,
            retrieval_session_factory=retrieval_session_factory,
        ),
    )

    builder.add_node(
        RESEARCH_EVIDENCE_EXTRACTOR,
        partial(
            research_evidence_extractor_node,
            llm_service=llm_service,
            settings=settings,
        ),
    )

    builder.add_node(
        RESEARCH_COVERAGE_EVALUATOR,
        partial(
            research_coverage_evaluator_node,
            llm_service=llm_service,
        ),
    )

    builder.add_node(
        RESEARCH_ROUTING,
        partial(
            research_routing_node,
            settings=settings,
        ),
    )

    builder.add_node(
        WRITER_PREPARE_EVIDENCE,
        partial(
            writer_prepare_evidence_node,
            settings=settings,
        ),
    )

    builder.add_node(
        WRITER_SYNTHESIS,
        partial(
            writer_synthesis_node,
            llm_service=llm_service,
        ),
    )

    builder.add_node(
        WRITER_CITATION_CHECKER,
        partial(
            writer_citation_checker_node,
            llm_service=llm_service,
        ),
    )

    builder.add_node(
        FINALIZE_REPORT,
        finalize_report_node,
    )

    #
    # Entry
    #

    builder.add_edge(
        START,
        INPUT_GUARDRAIL,
    )

    #
    # Input Guardrail
    #

    builder.add_conditional_edges(
        INPUT_GUARDRAIL,
        route_after_terminal_check,
        {
            "continue": RESEARCH_PLANNER,
            "terminal_error": END,
        },
    )

    #
    # Planner
    #

    builder.add_conditional_edges(
        RESEARCH_PLANNER,
        route_after_terminal_check,
        {
            "continue": RESEARCH_QUERY_GENERATOR,
            "terminal_error": END,
        },
    )

    #
    # Retrieval pipeline
    #

    builder.add_edge(
        RESEARCH_QUERY_GENERATOR,
        RESEARCH_SOURCE_SELECTION,
    )

    builder.add_conditional_edges(
        RESEARCH_SOURCE_SELECTION,
        route_after_source_selection,
        {
            "web_research": RESEARCH_WEB_SEARCH,
            "document_research": RESEARCH_RETRIEVAL,
            "terminal_error": END,
        },
    )

    builder.add_edge(
        RESEARCH_WEB_SEARCH,
        RESEARCH_SOURCE_PROCESSOR,
    )

    builder.add_edge(
        RESEARCH_SOURCE_PROCESSOR,
        RESEARCH_RETRIEVAL,
    )

    builder.add_edge(
        RESEARCH_RETRIEVAL,
        RESEARCH_EVIDENCE_EXTRACTOR,
    )

    builder.add_edge(
        RESEARCH_EVIDENCE_EXTRACTOR,
        RESEARCH_COVERAGE_EVALUATOR,
    )

    builder.add_edge(
        RESEARCH_COVERAGE_EVALUATOR,
        RESEARCH_ROUTING,
    )

    # Autonomous research loop
    # heart of the app

    builder.add_conditional_edges(
        RESEARCH_ROUTING,
        route_after_research,
        {
            "continue_research": RESEARCH_QUERY_GENERATOR,
            "write_report": WRITER_PREPARE_EVIDENCE,
            "terminal_error": END,
        },
    )

    # Writer pipeline
    
    builder.add_edge(
        WRITER_PREPARE_EVIDENCE,
        WRITER_SYNTHESIS,
    )

    builder.add_conditional_edges(
        WRITER_SYNTHESIS,
        route_after_terminal_check,
        {
            "continue": WRITER_CITATION_CHECKER,
            "terminal_error": END,
        },
    )

    builder.add_conditional_edges(
        WRITER_CITATION_CHECKER,
        route_after_terminal_check,
        {
            "continue": FINALIZE_REPORT,
            "terminal_error": END,
        },
    )

    # Finish

    builder.add_edge(
        FINALIZE_REPORT,
        END,
    )

    return builder.compile(
        name="deep_research_workflow",
    )

