import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb
from chromadb.utils import embedding_functions

# Initialize free models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Free, 384-dim
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Free LLM (Microsoft Phi-2 - runs locally)
try:
    llm = pipeline(
        "text-generation",
        model="microsoft/phi-2",
        device_map="auto",
        trust_remote_code=True,
        max_new_tokens=256
    )
    LLM_AVAILABLE = True
except:
    print("Phi-2 not available, using fallback")
    LLM_AVAILABLE = False

def embed_query_node(state: RAGState) -> RAGState:
    """Node 1: Embed the user query"""
    query = state["query"]
    
    # Generate embedding using free model
    embedding = embedding_model.encode(query).tolist()
    
    return {
        **state,
        "query_embedding": embedding
    }

def retrieve_docs_node(state: RAGState) -> RAGState:
    """Node 2: Retrieve relevant documents from ChromaDB"""
    query_embedding = state["query_embedding"]
    
    # Get or create collection
    try:
        collection = chroma_client.get_collection("astrophysics_papers")
    except:
        collection = chroma_client.create_collection(
            name="astrophysics_papers",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction()
        )
    
    # Search for similar documents
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )
    
    retrieved_docs = []
    if results['documents'] and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            retrieved_docs.append({
                "content": doc,
                "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                "distance": results['distances'][0][i] if results['distances'] else 1.0,
                "relevance_score": 1.0 - results['distances'][0][i] if results['distances'] else 0.5
            })
    
    return {
        **state,
        "retrieved_docs": retrieved_docs
    }

def rerank_docs_node(state: RAGState) -> RAGState:
    """Node 3: Rerank documents using cross-encoder (free)"""
    from sentence_transformers import CrossEncoder
    
    try:
        # Free cross-encoder for reranking
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    except:
        cross_encoder = None
    
    query = state["query"]
    docs = state["retrieved_docs"]
    
    if not docs or cross_encoder is None:
        return {**state, "reranked_docs": docs}
    
    # Prepare pairs for reranking
    pairs = [(query, doc["content"]) for doc in docs]
    scores = cross_encoder.predict(pairs)
    
    # Sort by reranking score
    for i, doc in enumerate(docs):
        doc["rerank_score"] = float(scores[i])
    
    reranked = sorted(docs, key=lambda x: x.get("rerank_score", 0), reverse=True)
    
    return {
        **state,
        "reranked_docs": reranked[:3]  # Keep top 3 after reranking
    }

def should_retrieve(state: RAGState) -> str:
    """Conditional edge: Check if we need to retry retrieval"""
    if not state.get("retrieved_docs") and state.get("retry_count", 0) < 2:
        return "retry"
    return "generate"

def generate_answer_node(state: RAGState) -> RAGState:
    """Node 4: Generate answer using free LLM"""
    query = state["query"]
    docs = state.get("reranked_docs", state.get("retrieved_docs", []))
    
    # Build context from retrieved documents
    context_parts = []
    sources = []
    
    for i, doc in enumerate(docs[:3]):
        context_parts.append(f"[Source {i+1}]: {doc['content']}")
        if doc.get('metadata', {}).get('source'):
            sources.append(doc['metadata']['source'])
    
    context = "\n\n".join(context_parts)
    
    # Calculate confidence based on relevance scores
    if docs:
        avg_relevance = np.mean([doc.get('relevance_score', 0.5) for doc in docs[:3]])
        confidence = min(0.95, 0.5 + (avg_relevance * 0.5))
    else:
        confidence = 0.3
    
    # Generate answer using free LLM
    if LLM_AVAILABLE:
        prompt = f"""You are an astrophysics expert. Answer based ONLY on the context below.

Context:
{context}

Question: {query}

Answer concisely based on the context:"""
        
        response = llm(prompt, max_new_tokens=256, temperature=0.1)
        answer = response[0]['generated_text'].replace(prompt, "").strip()
    else:
        # Fallback: extractive QA using sentence transformers
        answer = fallback_generate(query, docs)
    
    return {
        **state,
        "context": context,
        "answer": answer,
        "sources": list(set(sources)),
        "confidence": round(confidence, 3)
    }

def fallback_generate(query: str, docs: List[dict]) -> str:
    """Fallback when LLM not available - extractive QA"""
    from sentence_transformers import SentenceTransformer, util
    
    if not docs:
        return "No relevant documents found. Please upload astrophysics papers first."
    
    # Use embedding similarity to find best answer span
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_emb = model.encode(query, convert_to_tensor=True)
    
    best_doc = None
    best_score = 0
    
    for doc in docs:
        doc_emb = model.encode(doc['content'], convert_to_tensor=True)
        score = util.cos_sim(query_emb, doc_emb).item()
        if score > best_score:
            best_score = score
            best_doc = doc
    
    if best_doc and best_score > 0.5:
        # Extract relevant sentence (simple heuristic)
        sentences = best_doc['content'].split('.')
        relevant = []
        for sent in sentences:
            sent_emb = model.encode(sent, convert_to_tensor=True)
            if util.cos_sim(query_emb, sent_emb).item() > 0.6:
                relevant.append(sent.strip())
        
        if relevant:
            return ". ".join(relevant[:2]) + "."
        return best_doc['content'][:500]
    
    return "I found relevant context but cannot generate a specific answer. Please try a different question."

def error_handler_node(state: RAGState) -> RAGState:
    """Error handling node"""
    return {
        **state,
        "answer": "An error occurred while processing your request. Please try again.",
        "confidence": 0.0,
        "error": state.get("error", "Unknown error")
    }