from langgraph.graph import StateGraph, START, END
from .state import RAGState
from .nodes import (
    embed_query_node,
    retrieve_docs_node,
    rerank_docs_node,
    generate_answer_node,
    error_handler_node,
    should_retrieve
)

def build_rag_graph() -> StateGraph:
    """Build the LangGraph state machine for RAG"""
    
    # Create graph
    workflow = StateGraph(RAGState)
    
    # Add nodes
    workflow.add_node("embed_query", embed_query_node)
    workflow.add_node("retrieve_docs", retrieve_docs_node)
    workflow.add_node("rerank_docs", rerank_docs_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("error_handler", error_handler_node)
    
    # Add edges
    workflow.add_edge(START, "embed_query")
    workflow.add_edge("embed_query", "retrieve_docs")
    workflow.add_edge("retrieve_docs", "rerank_docs")
    
    # Conditional edge after rerank
    workflow.add_conditional_edges(
        "rerank_docs",
        should_retrieve,
        {
            "retry": "retrieve_docs",
            "generate": "generate_answer"
        }
    )
    
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("error_handler", END)
    
    return workflow.compile()