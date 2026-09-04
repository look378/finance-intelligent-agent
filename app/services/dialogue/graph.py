"""
LangGraph StateGraph builder for dialogue management.

Assembles the node graph, conditional edges, and memory checkpointer
into a compiled graph ready for invocation.
"""
import logging

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from app.services.dialogue.state import DialogueState

logger = logging.getLogger(__name__)


def build_dialogue_graph(
    intent_detector,
    tool_registry,
    retrieval_pipeline,
    llm_service,
    guardrail_service=None,
    graph_retrieval_service=None,
    slot_filler=None,
):
    """Build and compile the dialogue StateGraph.

    Args:
        intent_detector: Service implementing detect_with_confidence().
        tool_registry: ToolRegistry instance for Function Calling.
        retrieval_pipeline: Dict with optional "hybrid_search" key for RAG.
        llm_service: LLMServiceBase implementation for response generation.
        guardrail_service: Optional GuardrailService for input safety checks.
        graph_retrieval_service: Optional service for graph-based retrieval.
        slot_filler: 保留参数（槽位抽取由 slot_types 内置实现，未使用）。

    Returns:
        Compiled StateGraph with MemorySaver checkpointer.
    """
    from app.services.dialogue.nodes import NodeFactory

    factory = NodeFactory(
        intent_detector=intent_detector,
        slot_filler=slot_filler,
        tool_registry=tool_registry,
        retrieval_pipeline=retrieval_pipeline,
        llm_service=llm_service,
        guardrail_service=guardrail_service,
        graph_retrieval_service=graph_retrieval_service,
    )

    graph = StateGraph(DialogueState)

    # ── Nodes ────────────────────────────────────────────────────────────────
    graph.add_node("guardrail", factory.guardrail_node)
    graph.add_node("detect_intent", factory.detect_intent_node)
    graph.add_node("handle_switch", factory.handle_switch_node)
    graph.add_node("route_intent", factory.route_intent_node)
    graph.add_node("collect_slots", factory.collect_slots_node)
    graph.add_node("execute_tool", factory.execute_tool_node)
    graph.add_node("rag_lookup", factory.rag_lookup_node)
    graph.add_node("generate_response", factory.generate_response_node)
    graph.add_node("direct_response", factory.direct_response_node)

    # ── Fixed edges ──────────────────────────────────────────────────────────
    graph.add_edge(START, "guardrail")

    # After guardrail: skip intent detection when user is answering a slot prompt.
    graph.add_conditional_edges(
        "guardrail",
        factory.should_skip_intent,
        {
            "skip": "collect_slots",
            "full": "detect_intent",
        },
    )

    graph.add_edge("detect_intent", "handle_switch")
    graph.add_edge("handle_switch", "route_intent")

    # ── Conditional edges ────────────────────────────────────────────────────

    # After routing: fan out to the correct sub-pipeline.
    graph.add_conditional_edges(
        "route_intent",
        factory.route_by_intent,
        {
            "task": "collect_slots",
            "rag": "rag_lookup",
            "direct": "direct_response",
            "meta": "generate_response",
        },
    )

    # After slot collection: proceed to tool execution or ask for more info.
    graph.add_conditional_edges(
        "collect_slots",
        factory.check_slots,
        {
            "complete": "execute_tool",
            "missing": "generate_response",
        },
    )

    # ── Terminal edges ───────────────────────────────────────────────────────
    graph.add_edge("execute_tool", "generate_response")
    graph.add_edge("rag_lookup", "generate_response")
    graph.add_edge("direct_response", END)
    graph.add_edge("generate_response", END)

    # ── Compile with in-memory checkpointing ─────────────────────────────────
    checkpointer = MemorySaver()
    compiled = graph.compile(checkpointer=checkpointer)

    logger.info("Dialogue graph compiled successfully with %d nodes", len(graph.nodes))

    return compiled
