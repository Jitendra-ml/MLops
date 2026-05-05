from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Define state schema
class GraphState(TypedDict):
    input: str
    context: str
    output: str

# Retriever node
def retriever_node(state: GraphState) -> GraphState:
    query = state["input"]
    # Imagine this calls FAISS or Pinecone
    docs = f"Retrieved docs for: {query}"
    return {"context": docs, "input": query, "output": ""}

# LLM node
def llm_node(state: GraphState) -> GraphState:
    return {"output": f"Answer based on {state['context']}"}

# Build graph
graph = StateGraph(GraphState)
graph.add_node("retriever", retriever_node)
graph.add_node("llm", llm_node)

graph.add_edge(START, "retriever")
graph.add_edge("retriever", "llm")
graph.add_edge("llm", END)

app = graph.compile()

# Run
result = app.invoke({"input": "What is LangGraph?"})
print(result)
