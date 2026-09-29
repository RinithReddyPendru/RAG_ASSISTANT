from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import shutil
import time
from rag_core import ingest_documents, DATA_DIR, get_graph, embeddings
from rag_agent import agentic_rag
from semantic_cache import semantic_cache
from attribution_engine import attribute_citations

app = FastAPI(title="RAG Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from typing import Optional

class ChatRequest(BaseModel):
    message: str
    api_key: Optional[str] = None
    preferred_strategy: Optional[str] = "auto"
    include_context: bool = False

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        model_name = request.api_key if request.api_key else "gpt-oss:120b-cloud"
        preferred_strategy = request.preferred_strategy if request.preferred_strategy else "auto"
        t0 = time.perf_counter()
        
        # Check Semantic L1 Cache (sub-10ms lookup)
        q_vec = None
        try:
            q_vec = embeddings.embed_query(request.message)
            cached_res = semantic_cache.lookup(request.message, q_vec)
            if cached_res:
                print(f"---SEMANTIC CACHE HIT: Similarity={cached_res.get('cache_similarity')}---")
                return cached_res
        except Exception as e:
            print(f"Cache lookup warning: {e}")

        result = agentic_rag.invoke(
            {
                "question": request.message,
                "model_name": model_name,
                "preferred_strategy": preferred_strategy
            },
            {"recursion_limit": 20}
        )
        elapsed = round(time.perf_counter() - t0, 2)
        
        response = result.get("generation", "Sorry, I could not process your request.")
        strategy_used = result.get("strategy_selected", "vector")
        strategy_rationale = result.get("strategy_rationale", "")
        evaluation_score = result.get("evaluation_score", 0.0)
        evaluation_verdict = result.get("evaluation_verdict", "sufficient")
        loop_step = result.get("loop_step", 0)
        docs = result.get("documents", [])

        # Compute Sentence-Level Attribution & Citation Matrix
        attribution = attribute_citations(response, docs)
        
        return_data = {
            "response": response,
            "annotated_response": attribution.get("annotated_text", response),
            "attribution": attribution,
            "strategy_used": strategy_used,
            "strategy_rationale": strategy_rationale,
            "evaluation_score": evaluation_score,
            "evaluation_verdict": evaluation_verdict,
            "refinement_count": loop_step,
            "dw_rrf_weights": result.get("dw_rrf_weights"),
            "top_rerank_score": result.get("top_rerank_score"),
            "context": [doc.page_content for doc in docs],
            "latency": elapsed,
            "cache_hit": False
        }
        
        # Store in Semantic L1 Cache for recurring inquiries
        try:
            if q_vec:
                semantic_cache.store(request.message, q_vec, return_data, elapsed)
        except Exception as e:
            print(f"Cache store warning: {e}")
            
        return return_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import BackgroundTasks

@app.post("/api/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...), model_name: str = Form("llama3")):
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # Trigger ingestion after upload in the background
    background_tasks.add_task(ingest_documents, model_name, file_path)
    
    return {"message": f"Successfully uploaded {file.filename}, ingestion processing in background...", "chunks": "..."}

@app.get("/api/graph")
async def get_graph_data():
    """Returns knowledge graph nodes and edges for visualization in the UI."""
    try:
        graph = get_graph()
        nx_graph = getattr(graph, "_graph", graph)
        nodes = []
        edges = []
        if hasattr(nx_graph, "nodes"):
            for n in nx_graph.nodes():
                nodes.append({"id": str(n), "label": str(n), "title": str(n)})
            for u, v, data in nx_graph.edges(data=True):
                rel = data.get("relation", "related_to")
                edges.append({"from": str(u), "to": str(v), "label": str(rel), "arrows": "to"})
        return {"nodes": nodes, "edges": edges, "node_count": len(nodes), "edge_count": len(edges)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graph/path")
async def get_graph_path(source: str, target: str):
    """Finds the shortest multi-hop relational path connecting two entities across domains."""
    try:
        import networkx as nx
        graph = get_graph()
        nx_graph = getattr(graph, "_graph", graph)
        node_map = {str(n).lower().strip(): n for n in nx_graph.nodes()}
        src_node = node_map.get(source.lower().strip())
        tgt_node = node_map.get(target.lower().strip())
        
        if not src_node or not tgt_node:
            raise HTTPException(status_code=404, detail=f"Entity '{source if not src_node else target}' not found in Knowledge Graph.")
            
        undirected_g = nx_graph.to_undirected()
        if not nx.has_path(undirected_g, src_node, tgt_node):
            return {"found": False, "message": f"No relational path found between '{src_node}' and '{tgt_node}'."}
            
        path = nx.shortest_path(undirected_g, src_node, tgt_node)
        path_edges = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = nx_graph.get_edge_data(u, v) or nx_graph.get_edge_data(v, u) or {}
            rel = "related_to"
            if isinstance(edge_data, dict):
                first_val = next(iter(edge_data.values())) if edge_data else {}
                rel = first_val.get("relation", "relates to") if isinstance(first_val, dict) else edge_data.get("relation", "relates to")
            path_edges.append({"from": str(u), "to": str(v), "relation": str(rel)})
            
        narrative = " ➔ ".join([f"[{path[i]}] --({path_edges[i]['relation']})--> [{path[i+1]}]" for i in range(len(path_edges))])
        return {
            "found": True,
            "source": str(src_node),
            "target": str(tgt_node),
            "hops": len(path) - 1,
            "path_nodes": [str(p) for p in path],
            "path_edges": path_edges,
            "narrative": narrative
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ArenaRequest(BaseModel):
    message: str
    api_key: Optional[str] = None

@app.post("/api/arena")
async def arena_compare(request: ArenaRequest):
    """Executes the query across all 4 RAG paradigms concurrently for side-by-side comparison."""
    model_name = request.api_key if request.api_key else "gpt-oss:120b-cloud"
    strategies = [
        {"name": "Naive Dense Vector", "code": "vector", "icon": "🔍"},
        {"name": "Knowledge Graph", "code": "graph", "icon": "🕸️"},
        {"name": "Static Hybrid (DW-RRF)", "code": "hybrid", "icon": "🧬"},
        {"name": "Adaptive Multi-Strategy (Proposed)", "code": "auto", "icon": "🤖"}
    ]
    results = []
    for s in strategies:
        t0 = time.perf_counter()
        try:
            res = agentic_rag.invoke(
                {"question": request.message, "model_name": model_name, "preferred_strategy": s["code"]},
                {"recursion_limit": 20}
            )
            elapsed = round(time.perf_counter() - t0, 2)
            results.append({
                "name": s["name"],
                "icon": s["icon"],
                "strategy_code": s["code"],
                "strategy_selected": res.get("strategy_selected", s["code"]),
                "response": res.get("generation", ""),
                "evaluation_score": res.get("evaluation_score", 0.0),
                "top_rerank_score": res.get("top_rerank_score"),
                "dw_rrf_weights": res.get("dw_rrf_weights"),
                "latency": elapsed,
                "doc_count": len(res.get("documents", []))
            })
        except Exception as e:
            results.append({
                "name": s["name"],
                "icon": s["icon"],
                "strategy_code": s["code"],
                "error": str(e),
                "latency": round(time.perf_counter() - t0, 2)
            })
    return {"query": request.message, "comparisons": results}

@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": "Aura AI Adaptive RAG Backend",
        "platform": "Hugging Face Spaces (16GB RAM)",
        "endpoints": [
            "/api/chat",
            "/api/arena",
            "/api/graph",
            "/api/graph/path",
            "/api/upload"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)

