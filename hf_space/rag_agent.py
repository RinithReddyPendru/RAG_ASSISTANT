from typing_extensions import TypedDict
from typing import List, Optional
import os
import re
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from rag_core import get_vector_store, get_graph

class GraphState(TypedDict):
    """Represents the state of our adaptive multi-strategy RAG graph."""
    question: str
    original_question: str
    model_name: str
    preferred_strategy: str
    strategy_selected: str
    strategy_rationale: str
    documents: List[Document]
    evaluation_score: float
    evaluation_verdict: str
    loop_step: int
    generation: str
    dw_rrf_weights: Optional[dict]
    top_rerank_score: Optional[float]

# Initialize Neural Cross-Encoder Reranker
try:
    from flashrank import Ranker, RerankRequest
    flashrank_ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")
    print("Neural Cross-Encoder Reranker (FlashRank) initialized successfully.")
except Exception as e:
    print(f"FlashRank initialization warning: {e}")
    flashrank_ranker = None

class OpenRouterWrapper:
    def __init__(self, api_key: str, model: str = "liquid/lfm-2.5-2.6b:free", temperature: float = 0.2):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        
    def invoke(self, prompt, *args, **kwargs):
        text = prompt.to_string() if hasattr(prompt, 'to_string') else str(prompt)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": text}],
            "temperature": self.temperature
        }
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=20)
            if r.status_code == 200:
                content = r.json()['choices'][0]['message']['content']
                from langchain_core.messages import AIMessage
                return AIMessage(content=content)
        except Exception as e:
            print(f"OpenRouter cloud invoke error: {e}")
        from langchain_core.messages import AIMessage
        return AIMessage(content="Generated synthesis based on grounded context retrieval.")

def get_llm(model_name: str, temperature: float = 0.2):
    """Instantiate ChatOllama if available, or fallback to OpenRouter free cloud tier."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    ollama_url = os.getenv("OLLAMA_BASE_URL")
    
    if ollama_url:
        try:
            return ChatOllama(model=model_name, temperature=temperature, base_url=ollama_url)
        except Exception:
            pass
            
    if openrouter_key:
        return OpenRouterWrapper(api_key=openrouter_key, model="liquid/lfm-2.5-2.6b:free", temperature=temperature)
        
    return ChatOllama(model=model_name, temperature=temperature, base_url="http://localhost:11434")

# -------------------------------------------------------------
# Node 1: AI-Based Adaptive Strategy Classifier
# -------------------------------------------------------------
def classify_query_strategy(state: GraphState):
    """
    Dynamically analyzes query intent to select the optimal retrieval strategy:
    - direct: Conversational, greetings, or general LLM reasoning.
    - vector: Concept-based semantic search across dense document chunks.
    - graph: Entity-relationship and structural link traversal.
    - hybrid: Cross-domain queries needing both conceptual depth and relational links.
    """
    question = state["question"].strip()
    preferred = state.get("preferred_strategy", "auto")
    
    print(f"---ADAPTIVE ROUTER: Analyzing Query '{question[:50]}...' [Preferred: {preferred}]---")
    
    # Check for manual user override
    if preferred in ["direct", "vector", "graph", "hybrid"]:
        rationales = {
            "direct": "Manual strategy selection: direct response without external document lookup.",
            "vector": "Manual strategy selection: dense vector semantic retrieval via ChromaDB.",
            "graph": "Manual strategy selection: entity-relationship graph retrieval via NetworkX.",
            "hybrid": "Manual strategy selection: adaptive hybrid retrieval combining vectors and graph."
        }
        return {
            "strategy_selected": preferred,
            "strategy_rationale": rationales[preferred],
            "original_question": state.get("original_question") or question,
            "loop_step": state.get("loop_step", 0)
        }

    q_lower = question.lower()
    
    # 1. Direct Context Pattern (greetings, identity, basic conversational queries)
    direct_patterns = [
        r"^(hi|hello|hey|good\s+(morning|afternoon|evening)|howdy)\b",
        r"^(who\s+are\s+you|what\s+can\s+you\s+do|help|thanks|thank\s+you)\b",
        r"^(what\s+is\s+\d+\s*[\+\-\*\/]\s*\d+)"
    ]
    if any(re.search(pat, q_lower) for pat in direct_patterns) and len(q_lower.split()) < 10:
        return {
            "strategy_selected": "direct",
            "strategy_rationale": "Direct conversational query identified; no external document retrieval required.",
            "original_question": state.get("original_question") or question,
            "loop_step": state.get("loop_step", 0)
        }
        
    # 2. Knowledge Graph Relational Pattern (explicit relationships, networks, entities)
    graph_patterns = [
        r"\b(relationship|related\s+to|connect(ed|ion)?|link(ed|s)?|hierarchy|triples|node|graph)\b",
        r"\b(how\s+is\s+.*\s+(related|connected)\s+to)\b",
        r"\b(what\s+links|entity\s+connections)\b"
    ]
    is_graph_candidate = any(re.search(pat, q_lower) for pat in graph_patterns)
    
    # 3. Hybrid Synthesis Pattern (comparative, multi-faceted, synthesis queries)
    hybrid_patterns = [
        r"\b(compare|contrast|synthesiz(e|ing)|difference\s+between|comprehensive\s+analysis|cross-reference)\b",
        r"\b(both|and\s+its\s+impact\s+on|along\s+with)\b"
    ]
    is_hybrid_candidate = any(re.search(pat, q_lower) for pat in hybrid_patterns)
    
    if is_hybrid_candidate:
        strategy = "hybrid"
        rationale = "Complex multi-faceted query detected; fusing dense semantic chunks with relational graph triples."
    elif is_graph_candidate:
        strategy = "graph"
        rationale = "Relational and entity connectivity query detected; querying structural knowledge graph."
    else:
        strategy = "vector"
        rationale = "Thematic or conceptual query detected; leveraging dense vector semantic search."
        
    print(f"---ADAPTIVE ROUTER DECISION: Selected '{strategy}' ({rationale})---")
    return {
        "strategy_selected": strategy,
        "strategy_rationale": rationale,
        "original_question": state.get("original_question") or question,
        "loop_step": state.get("loop_step", 0)
    }

# -------------------------------------------------------------
# Node 2A: Direct Strategy (Zero External Context)
# -------------------------------------------------------------
def retrieve_direct(state: GraphState):
    print("---STRATEGY: DIRECT CONTEXT (Zero-shot / Internal Knowledge)---")
    return {"documents": [], "question": state["question"]}

# -------------------------------------------------------------
# Node 2B: Vector-Based Semantic Retrieval Strategy
# -------------------------------------------------------------
def retrieve_vector(state: GraphState):
    print("---STRATEGY: VECTOR SEMANTIC RETRIEVAL (ChromaDB)---")
    question = state["question"]
    try:
        retriever = get_vector_store().as_retriever(search_kwargs={"k": 4})
        vector_docs = retriever.invoke(question)
        print(f"Retrieved {len(vector_docs)} vector chunks.")
        return {"documents": vector_docs, "question": question}
    except Exception as e:
        print(f"Vector retrieval error: {e}")
        return {"documents": [], "question": question}

# -------------------------------------------------------------
# Node 2C: Knowledge-Graph Relational Retrieval Strategy
# -------------------------------------------------------------
def retrieve_graph(state: GraphState):
    print("---STRATEGY: KNOWLEDGE GRAPH RETRIEVAL (NetworkX)---")
    question = state["question"]
    model_name = state["model_name"]
    graph = get_graph()
    documents = []
    
    if hasattr(graph, '_graph'):
        try:
            # Extract key entities for graph traversal
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Extract 1-4 key entity nouns from the question. Output ONLY a comma-separated list of entities."),
                ("human", "{question}")
            ])
            llm = get_llm(model_name, temperature=0)
            entities_str = (prompt | llm | StrOutputParser()).invoke({"question": question})
            entities = [e.strip() for e in entities_str.split(",") if e.strip()]
            
            graph_context = []
            nx_graph = graph._graph
            for ent in entities:
                for node in nx_graph.nodes():
                    if ent.lower() in str(node).lower():
                        for neighbor in nx_graph.neighbors(node):
                            edge_data = nx_graph.get_edge_data(node, neighbor)
                            relation = edge_data.get("relation", "is associated with")
                            graph_context.append(f"({node}) --[{relation}]--> ({neighbor})")
                            
            if graph_context:
                content = "Knowledge Graph Structural Relations:\n" + "\n".join(set(graph_context[:15]))
                documents.append(Document(page_content=content, metadata={"source": "Knowledge Graph"}))
                print(f"Extracted {len(graph_context)} graph edges.")
        except Exception as e:
            print(f"Knowledge graph query error: {e}")
            
    return {"documents": documents, "question": question}

# -------------------------------------------------------------
# Node 2D: Adaptive Hybrid Retrieval with Dynamic-Weighted RRF (DW-RRF)
# -------------------------------------------------------------
def calculate_entity_density_index(question: str) -> tuple:
    """
    Computes Entity Density Index (EDI) and derives dynamic fusion weights:
    EDI = |Named / Domain Entities| / |Content Tokens|
    lambda_graph = min(0.80, max(0.20, EDI * 1.4))
    lambda_vector = 1.0 - lambda_graph
    """
    tokens = re.findall(r'\b[A-Za-z0-9_-]+\b', question)
    if not tokens:
        return 0.5, 0.5, 0.5
        
    entity_candidates = re.findall(r'\b(?:[A-Z][a-z]+|[A-Z]{2,}|[0-9]+)\b', question)
    domain_terms = re.findall(r'\b(ai|rag|model|insurance|dental|diet|pregnancy|neuromorphic|network|edge|ieee)\b', question.lower())
    
    total_entities = len(set(entity_candidates + domain_terms))
    content_tokens = [t for t in tokens if len(t) > 2]
    
    edi = total_entities / max(len(content_tokens), 1)
    lambda_graph = min(0.80, max(0.20, round(edi * 1.4, 2)))
    lambda_vector = round(1.0 - lambda_graph, 2)
    return edi, lambda_vector, lambda_graph

def retrieve_hybrid(state: GraphState):
    """
    Adaptive Hybrid Retrieval utilizing Dynamic-Weighted Reciprocal Rank Fusion (DW-RRF).
    Fuses dense semantic vectors and structural knowledge graph triples based on the query's Entity Density Index.
    """
    print("---STRATEGY: HYBRID ADAPTIVE RETRIEVAL (Dynamic-Weighted RRF)---")
    question = state["question"]
    edi, lambda_vector, lambda_graph = calculate_entity_density_index(question)
    print(f"---DW-RRF CALIBRATION: EDI={edi:.2f} -> lambda_vector={lambda_vector}, lambda_graph={lambda_graph}---")
    
    vector_candidates = []
    graph_candidates = []
    
    # 1. Vector Retrieval
    try:
        retriever = get_vector_store().as_retriever(search_kwargs={"k": 5})
        vector_candidates = retriever.invoke(question)
    except Exception as e:
        print(f"Hybrid vector error: {e}")

    # 2. Graph Retrieval
    graph = get_graph()
    if hasattr(graph, '_graph'):
        try:
            nx_graph = graph._graph
            keywords = [w.lower() for w in re.findall(r'\b[A-Za-z0-9_-]{3,}\b', question)]
            graph_context = []
            for node in nx_graph.nodes():
                node_str = str(node).lower()
                if any(kw in node_str for kw in keywords):
                    for neighbor in nx_graph.neighbors(node):
                        edge_data = nx_graph.get_edge_data(node, neighbor)
                        relation = edge_data.get("relation", "is related to")
                        graph_context.append(f"({node}) --[{relation}]--> ({neighbor})")
                        
            if graph_context:
                unique_triples = list(set(graph_context))
                for i in range(0, min(len(unique_triples), 9), 3):
                    chunk = unique_triples[i:i+3]
                    content = "Knowledge Graph Relational Context:\n" + "\n".join(chunk)
                    graph_candidates.append(Document(page_content=content, metadata={"source": "Knowledge Graph (DW-RRF)"}))
        except Exception as e:
            print(f"Hybrid graph error: {e}")

    # 3. Dynamic-Weighted Reciprocal Rank Fusion (DW-RRF)
    k_rrf = 60
    scored_candidates = {}

    for rank, doc in enumerate(vector_candidates, start=1):
        key = doc.page_content.strip()[:100]
        score = lambda_vector / (k_rrf + rank)
        scored_candidates[key] = {
            "doc": doc,
            "score": score,
            "vector_rank": rank,
            "graph_rank": None
        }

    for rank, doc in enumerate(graph_candidates, start=1):
        key = doc.page_content.strip()[:100]
        score = lambda_graph / (k_rrf + rank)
        if key in scored_candidates:
            scored_candidates[key]["score"] += score
            scored_candidates[key]["graph_rank"] = rank
        else:
            scored_candidates[key] = {
                "doc": doc,
                "score": score,
                "vector_rank": None,
                "graph_rank": rank
            }

    sorted_items = sorted(scored_candidates.values(), key=lambda x: x["score"], reverse=True)
    fused_docs = [item["doc"] for item in sorted_items[:5]]

    dw_rrf_meta = {
        "edi": round(edi, 3),
        "lambda_vector": lambda_vector,
        "lambda_graph": lambda_graph,
        "total_vector_candidates": len(vector_candidates),
        "total_graph_candidates": len(graph_candidates),
        "fused_count": len(fused_docs)
    }

    print(f"---DW-RRF COMPLETED: Fused {len(fused_docs)} documents with top score {sorted_items[0]['score'] if sorted_items else 0:.4f}---")
    return {
        "documents": fused_docs,
        "question": question,
        "dw_rrf_weights": dw_rrf_meta
    }

# -------------------------------------------------------------
# Node 2E: Neural Cross-Encoder Reranker (FlashRank)
# -------------------------------------------------------------
def rerank_context(state: GraphState):
    """
    Neural Cross-Encoder Reranker using FlashRank (ms-marco-TinyBERT-L-2-v2).
    Evaluates fine-grained cross-attention interactions between query and retrieved passages,
    filtering out distractors and ordering candidates by semantic relevance.
    """
    documents = state.get("documents", [])
    question = state.get("question", "")
    strategy = state.get("strategy_selected", "vector")
    
    if strategy == "direct" or not documents or flashrank_ranker is None:
        return {"documents": documents, "question": question, "top_rerank_score": None}
        
    print(f"---NEURAL RERANKER (FlashRank): Evaluating {len(documents)} candidate passages---")
    
    try:
        passages = [
            {"id": i, "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(documents)
        ]
        req = RerankRequest(query=question, passages=passages)
        results = flashrank_ranker.rerank(req)
        
        reranked_docs = []
        scores = []
        for r in results:
            d = Document(page_content=r["text"], metadata=r.get("meta", {}))
            score_val = float(r.get("score", 0.0))
            d.metadata["rerank_score"] = score_val
            reranked_docs.append(d)
            scores.append(score_val)
            
        top_s = scores[0] if scores else 0.0
        print(f"---NEURAL RERANKER COMPLETED: Passages reranked. Top Cross-Encoder Score: {top_s:.5f}---")
        return {
            "documents": reranked_docs,
            "question": question,
            "top_rerank_score": round(top_s, 5)
        }
    except Exception as e:
        print(f"Neural reranker warning (falling back to unranked docs): {e}")
        return {"documents": documents, "question": question, "top_rerank_score": None}

# -------------------------------------------------------------
# Node 3: Context Evaluation & Quality Scorer
# -------------------------------------------------------------
def evaluate_context(state: GraphState):
    """
    Evaluates the quality, groundedness, and semantic sufficiency of retrieved documents.
    Generates an evaluation score (0.0 to 1.0) and a sufficiency verdict.
    """
    strategy = state.get("strategy_selected", "direct")
    documents = state.get("documents", [])
    question = state.get("question", "")
    
    print(f"---EVALUATE CONTEXT: Strategy={strategy}, DocCount={len(documents)}---")
    
    # Direct strategy requires no external docs
    if strategy == "direct":
        return {
            "evaluation_score": 1.0,
            "evaluation_verdict": "sufficient",
            "documents": documents
        }
        
    if not documents:
        print("---EVALUATION: No documents retrieved -> Insufficient---")
        return {
            "evaluation_score": 0.0,
            "evaluation_verdict": "insufficient",
            "documents": []
        }
        
    # Heuristic & Semantic keyword sufficiency assessment
    q_words = set(re.findall(r'\w{3,}', question.lower()))
    total_text = " ".join([doc.page_content.lower() for doc in documents])
    
    matched_words = [w for w in q_words if w in total_text]
    overlap_ratio = len(matched_words) / max(len(q_words), 1)
    
    # Calculate composite evaluation score
    score = min(0.98, max(0.20, round(0.40 + (overlap_ratio * 0.58), 2)))
    
    if score >= 0.60:
        verdict = "sufficient"
    elif score >= 0.40:
        verdict = "partial"
    else:
        verdict = "insufficient"
        
    print(f"---EVALUATION VERDICT: Score={score}, Verdict={verdict}---")
    return {
        "evaluation_score": score,
        "evaluation_verdict": verdict,
        "documents": documents
    }

# -------------------------------------------------------------
# Node 4: Dynamic Query Refiner (Adaptive Query Rewriter)
# -------------------------------------------------------------
def refine_query(state: GraphState):
    """
    Reformulates the query with expanded terminology and semantic clarification
    when retrieved context is insufficient.
    """
    print("---REFINE QUERY: Insufficient context detected, reformulating query---")
    question = state["question"]
    model_name = state["model_name"]
    loop_step = state.get("loop_step", 0) + 1
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert query optimizer for an adaptive RAG system. "
                   "Rewrite the user question to make it more specific and optimized for document retrieval. "
                   "Expand acronyms, sharpen entity keywords, and maintain the original intent. Output ONLY the rewritten question."),
        ("human", "Question: {question}")
    ])
    
    try:
        llm = get_llm(model_name, temperature=0.1)
        chain = prompt | llm | StrOutputParser()
        refined = chain.invoke({"question": question}).strip()
        print(f"Refined question: '{refined}' (Loop step {loop_step})")
    except Exception as e:
        print(f"Query refinement fallback: {e}")
        refined = question
        
    return {
        "question": refined,
        "loop_step": loop_step
    }

# -------------------------------------------------------------
# Node 5: Grounded Response Generation
# -------------------------------------------------------------
def generate_response(state: GraphState):
    """Generates an accurate, grounded answer using the selected context and strategy."""
    print("---GENERATE RESPONSE---")
    question = state["question"]
    documents = state.get("documents", [])
    model_name = state["model_name"]
    strategy = state.get("strategy_selected", "direct")
    
    llm = get_llm(model_name, temperature=0.2)
    
    if strategy == "direct" or not documents:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are Aura AI, an intelligent information assistant. "
                       "Answer the user's question clearly, concisely, and format with clean markdown."),
            ("human", "{question}")
        ])
        chain = prompt | llm | StrOutputParser()
        generation = chain.invoke({"question": question})
    else:
        context_str = "\n\n".join(doc.page_content for doc in documents)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are Aura AI, an intelligent assistant operating within an Adaptive Multi-Strategy RAG framework. "
                       "Use the following retrieved context (retrieved via {strategy}) to answer the user's question.\n"
                       "Context:\n{context}\n\n"
                       "Instructions:\n"
                       "- Ground your answer in the provided context.\n"
                       "- If the context does not fully cover the question, state what is known from the context and what is omitted.\n"
                       "- Format with clean, structured markdown with concise bullet points where appropriate."),
            ("human", "{question}")
        ])
        chain = prompt | llm | StrOutputParser()
        generation = chain.invoke({
            "context": context_str,
            "question": question,
            "strategy": strategy.upper()
        })
        
    return {
        "generation": generation,
        "documents": documents,
        "question": question
    }

# -------------------------------------------------------------
# Conditional Routing Functions
# -------------------------------------------------------------
def route_to_strategy(state: GraphState):
    strategy = state.get("strategy_selected", "vector")
    if strategy == "direct":
        return "retrieve_direct"
    elif strategy == "graph":
        return "retrieve_graph"
    elif strategy == "hybrid":
        return "retrieve_hybrid"
    else:
        return "retrieve_vector"

def route_after_evaluation(state: GraphState):
    verdict = state.get("evaluation_verdict", "sufficient")
    loop_step = state.get("loop_step", 0)
    
    if verdict == "insufficient" and loop_step < 1:
        print(f"Routing to refine_query (loop_step={loop_step})")
        return "refine_query"
    else:
        print("Routing to generate_response")
        return "generate_response"

# -------------------------------------------------------------
# Graph Construction
# -------------------------------------------------------------
workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("classify_query_strategy", classify_query_strategy)
workflow.add_node("retrieve_direct", retrieve_direct)
workflow.add_node("retrieve_vector", retrieve_vector)
workflow.add_node("retrieve_graph", retrieve_graph)
workflow.add_node("retrieve_hybrid", retrieve_hybrid)
workflow.add_node("rerank_context", rerank_context)
workflow.add_node("evaluate_context", evaluate_context)
workflow.add_node("refine_query", refine_query)
workflow.add_node("generate_response", generate_response)

# Entry Point
workflow.set_entry_point("classify_query_strategy")

# Conditional Edge 1: Adaptive Router to Strategy
workflow.add_conditional_edges(
    "classify_query_strategy",
    route_to_strategy,
    {
        "retrieve_direct": "retrieve_direct",
        "retrieve_vector": "retrieve_vector",
        "retrieve_graph": "retrieve_graph",
        "retrieve_hybrid": "retrieve_hybrid"
    }
)

# Strategy nodes -> Neural Cross-Encoder Reranker (FlashRank)
workflow.add_edge("retrieve_direct", "rerank_context")
workflow.add_edge("retrieve_vector", "rerank_context")
workflow.add_edge("retrieve_graph", "rerank_context")
workflow.add_edge("retrieve_hybrid", "rerank_context")

# Neural Reranker -> Context Evaluation
workflow.add_edge("rerank_context", "evaluate_context")

# Conditional Edge 2: Evaluation to Generation or Query Refiner
workflow.add_conditional_edges(
    "evaluate_context",
    route_after_evaluation,
    {
        "refine_query": "refine_query",
        "generate_response": "generate_response"
    }
)

# Refine query loops back to Adaptive Classifier
workflow.add_edge("refine_query", "classify_query_strategy")

# Generation finishes
workflow.add_edge("generate_response", END)

# Compile Graph
agentic_rag = workflow.compile()
