from typing import TypedDict, List, Annotated, Optional
import operator

class RAGState(TypedDict):
    """State schema for LangGraph RAG workflow"""
    query: str
    query_embedding: Optional[List[float]]
    retrieved_docs: Annotated[List[dict], operator.add]
    reranked_docs: List[dict]
    context: str
    answer: str
    sources: List[str]
    confidence: float
    retry_count: int
    error: Optional[str]