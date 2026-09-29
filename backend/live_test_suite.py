"""
Live Verification & Accuracy Test Suite for Adaptive Multi-Strategy RAG
Executes end-to-end evaluation queries across Direct, Vector, Graph, and Hybrid paradigms.
Calculates:
- Routing Decision Accuracy (%)
- Fact Recall / Knowledge Extraction Hit Rate (%)
- Grounded Attribution Rate (%)
- End-to-End System Pass Rate (%)
- Average Latency (s) and Semantic Cache Hit Performance
"""

import sys
import os
import time
import json
import urllib.request
import urllib.error

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

API_URL = "http://127.0.0.1:8000/api/chat"
MODEL_NAME = "gpt-oss:120b-cloud"

TEST_CASES = [
    # Category 1: Direct Context / Conversational Reasoning (Zero external retrieval needed)
    {
        "id": "TC_DIR_01",
        "category": "Direct Context",
        "query": "Hello Aura, what is your role and what can you help me with?",
        "expected_strategy": ["direct"],
        "required_facts": ["aura", "assistant", "document", "retriev"]
    },
    {
        "id": "TC_DIR_02",
        "category": "Direct Context",
        "query": "Define the concept of Retrieval-Augmented Generation (RAG) in AI.",
        "expected_strategy": ["direct"],
        "required_facts": ["retrieval", "generation", "knowledge", "model"]
    },
    {
        "id": "TC_DIR_03",
        "category": "Direct Context",
        "query": "Can you explain what a neural cross-encoder does conceptually?",
        "expected_strategy": ["direct"],
        "required_facts": ["cross", "rerank", "query", "passage"]
    },
    {
        "id": "TC_DIR_04",
        "category": "Direct Context",
        "query": "Good morning! Are you ready to analyze research papers?",
        "expected_strategy": ["direct"],
        "required_facts": ["ready", "help", "assist"]
    },

    # Category 2: Vector Semantic Retrieval (Dense Thematic Chunks)
    {
        "id": "TC_VEC_05",
        "category": "Vector Semantic",
        "query": "What are the six major structural gaps in Indian dental insurance?",
        "expected_strategy": ["vector"],
        "required_facts": ["out-of-pocket", "exclusion", "pricing", "tariff", "gap"]
    },
    {
        "id": "TC_VEC_06",
        "category": "Vector Semantic",
        "query": "What is the ABDM digital health ecosystem and its role in Indian healthcare?",
        "expected_strategy": ["vector"],
        "required_facts": ["abdm", "ayushman bharat", "digital", "standard"]
    },
    {
        "id": "TC_VEC_07",
        "category": "Vector Semantic",
        "query": "How does all-MiniLM-L6-v2 generate embeddings stored in ChromaDB?",
        "expected_strategy": ["vector"],
        "required_facts": ["embedding", "dense", "chromadb", "vector"]
    },
    {
        "id": "TC_VEC_08",
        "category": "Vector Semantic",
        "query": "What are the trimester-specific dietary macronutrient guidelines for pregnancy?",
        "expected_strategy": ["vector"],
        "required_facts": ["trimester", "diet", "protein", "energy", "maternal"]
    },

    # Category 3: Knowledge Graph Relational Retrieval (Entity Links & Traversal)
    {
        "id": "TC_GRP_09",
        "category": "Knowledge Graph",
        "query": "Which rural healthcare facilities benefit from Tiered Rural Micro-Subsidies?",
        "expected_strategy": ["graph", "vector"],
        "required_facts": ["primary health centres", "phc", "subsid"]
    },
    {
        "id": "TC_GRP_10",
        "category": "Knowledge Graph",
        "query": "What neuron model in spiking neural networks integrates incoming spikes to reach a membrane threshold?",
        "expected_strategy": ["graph", "vector"],
        "required_facts": ["leaky integrate-and-fire", "lif", "membrane", "threshold"]
    },
    {
        "id": "TC_GRP_11",
        "category": "Knowledge Graph",
        "query": "What physiological condition in maternal healthcare is mitigated by iron and folic acid supplementation?",
        "expected_strategy": ["graph", "vector"],
        "required_facts": ["anemia", "folic acid", "iron", "gestational"]
    },
    {
        "id": "TC_GRP_12",
        "category": "Knowledge Graph",
        "query": "What architecture orchestrates the Adaptive Strategy Router, and what strategies does it select?",
        "expected_strategy": ["graph", "hybrid", "vector"],
        "required_facts": ["langgraph", "router", "direct", "vector", "graph", "hybrid"]
    },

    # Category 4: Adaptive Hybrid DW-RRF (Complex Relational & Cross-Domain Synthesis)
    {
        "id": "TC_HYB_13",
        "category": "Adaptive Hybrid",
        "query": "How does the Entity Density Index balance vector weights and graph weights during DW-RRF?",
        "expected_strategy": ["hybrid", "graph"],
        "required_facts": ["edi", "density", "lambda", "fusion", "weight"]
    },
    {
        "id": "TC_HYB_14",
        "category": "Adaptive Hybrid",
        "query": "Compare third-party administrator adjudication protocols with autonomous smart contract settlement.",
        "expected_strategy": ["hybrid"],
        "required_facts": ["tpa", "smart contract", "adjudication", "settlement"]
    },
    {
        "id": "TC_HYB_15",
        "category": "Adaptive Hybrid",
        "query": "How do memristive crossbar arrays achieve energy efficiency compared to standard vector retrieval?",
        "expected_strategy": ["hybrid", "vector"],
        "required_facts": ["memristive", "energy", "crossbar", "retrieval"]
    },
    {
        "id": "TC_HYB_16",
        "category": "Adaptive Hybrid",
        "query": "What component triggers the self-correcting refinement loop in Aura AI when retrieved context is judged insufficient?",
        "expected_strategy": ["hybrid", "vector"],
        "required_facts": ["scorer", "sufficiency", "refinement", "loop"]
    }
]

def run_test_suite():
    print("=" * 85)
    print("      AURA AI: LIVE ADAPTIVE MULTI-STRATEGY RAG TEST SUITE")
    print(f"Target URL: {API_URL} | Model: {MODEL_NAME}")
    print(f"Total Test Cases: {len(TEST_CASES)} across 4 Strategy Domains")
    print("=" * 85)

    results = []
    
    for idx, tc in enumerate(TEST_CASES, 1):
        print(f"\n[{idx}/{len(TEST_CASES)}] Executing {tc['id']} ({tc['category']}):")
        print(f"    Query: \"{tc['query']}\"")
        
        payload = {
            "message": tc["query"],
            "api_key": MODEL_NAME,
            "preferred_strategy": "auto"
        }
        
        req = urllib.request.Request(
            API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                elapsed = time.perf_counter() - t0
                data = json.loads(resp.read().decode("utf-8"))
                
                resp_text = (data.get("response") or "").lower()
                strategy_used = data.get("strategy_used", "unknown")
                eval_score = float(data.get("evaluation_score") or 0.0)
                cache_hit = bool(data.get("cache_hit", False))
                
                # Check Strategy Router Match
                router_match = strategy_used in tc["expected_strategy"]
                
                # Check Fact Extraction Recall
                matched_facts = [f for f in tc["required_facts"] if f.lower() in resp_text]
                fact_recall = round(len(matched_facts) / max(len(tc["required_facts"]), 1), 3)
                
                # Attribution Groundedness
                attribution_data = data.get("attribution", {})
                attribution_rate = float(attribution_data.get("attribution_score", eval_score))
                
                # Pass / Partial / Fail Determination
                if fact_recall >= 0.60 and attribution_rate >= 0.50:
                    verdict = "PASS"
                elif fact_recall >= 0.40:
                    verdict = "PARTIAL"
                else:
                    verdict = "FAIL"
                
                print(f"    -> Selected: [{strategy_used}] (Expected: {tc['expected_strategy']}) | Router: {'✓' if router_match else '✗'}")
                print(f"    -> Fact Recall: {int(fact_recall*100)}% ({len(matched_facts)}/{len(tc['required_facts'])}) | Grounded: {int(attribution_rate*100)}% | Time: {elapsed:.2f}s | Verdict: [{verdict}]")
                
                results.append({
                    "id": tc["id"],
                    "category": tc["category"],
                    "query": tc["query"],
                    "selected_strategy": strategy_used,
                    "router_match": router_match,
                    "fact_recall": fact_recall,
                    "attribution_rate": attribution_rate,
                    "latency": round(elapsed, 2),
                    "cache_hit": cache_hit,
                    "verdict": verdict
                })
        except Exception as e:
            print(f"    -> ERROR executing {tc['id']}: {e}")
            results.append({
                "id": tc["id"],
                "category": tc["category"],
                "query": tc["query"],
                "selected_strategy": "error",
                "router_match": False,
                "fact_recall": 0.0,
                "attribution_rate": 0.0,
                "latency": 0.0,
                "cache_hit": False,
                "verdict": "ERROR"
            })

    # Summary Statistics
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["verdict"] == "PASS")
    partial_tests = sum(1 for r in results if r["verdict"] == "PARTIAL")
    failed_tests = sum(1 for r in results if r["verdict"] in ["FAIL", "ERROR"])
    
    router_correct = sum(1 for r in results if r["router_match"])
    avg_fact_recall = sum(r["fact_recall"] for r in results) / total_tests
    avg_attribution = sum(r["attribution_rate"] for r in results) / total_tests
    avg_latency = sum(r["latency"] for r in results) / total_tests
    
    # Working Accuracy Metric: (Passed + 0.5 * Partial) / Total
    working_accuracy = round(((passed_tests + 0.5 * partial_tests) / total_tests) * 100, 1)
    router_accuracy = round((router_correct / total_tests) * 100, 1)

    print("\n" + "=" * 85)
    print("                 LIVE TEST SUITE RESULTS SUMMARY")
    print("=" * 85)
    print(f"Total Test Cases Executed      : {total_tests}")
    print(f"PASSED (Full Recall & Ground)  : {passed_tests} ({passed_tests/total_tests*100:.1f}%)")
    print(f"PARTIAL (Moderate Recall)     : {partial_tests} ({partial_tests/total_tests*100:.1f}%)")
    print(f"FAILED / ERROR                : {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
    print("-" * 85)
    print(f"🎯 OVERALL WORKING ACCURACY     : {working_accuracy}%")
    print(f"🤖 ROUTER CLASSIFICATION ACCURACY: {router_accuracy}%")
    print(f"📚 AVERAGE FACT EXTRACTION RECALL : {avg_fact_recall*100:.1f}%")
    print(f"🛡️ AVERAGE GROUNDED ATTRIBUTION  : {avg_attribution*100:.1f}%")
    print(f"⏱️ AVERAGE RESPONSE LATENCY      : {avg_latency:.2f}s")
    print("=" * 85)

    # Save Results to JSON and CSV
    out_dir = os.path.join(os.path.dirname(__file__), "..", "eval_results")
    os.makedirs(out_dir, exist_ok=True)
    
    json_path = os.path.join(out_dir, "live_test_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "partial": partial_tests,
                "failed": failed_tests,
                "working_accuracy_pct": working_accuracy,
                "router_accuracy_pct": router_accuracy,
                "avg_fact_recall_pct": round(avg_fact_recall * 100, 1),
                "avg_attribution_pct": round(avg_attribution * 100, 1),
                "avg_latency_sec": round(avg_latency, 2)
            },
            "cases": results
        }, f, indent=2)
    print(f"\n[OK] Detailed test report saved to: {json_path}")

if __name__ == "__main__":
    run_test_suite()
