"""
Aura AI — Final 5-Metric RAG Evaluation Suite
==============================================
1. BERTScore F1          — Semantic correctness vs ground truth
2. Answer Relevance      — Is the answer on-topic to the question
3. Answer Correctness    — Accuracy of answer vs ground truth (cosine)
4. Hallucination Score   — Detects made-up content via NLI model
5. Query Rewrite         — Agentic robustness on vague/poor questions

Run: python rag_eval_5metrics.py
"""

# ── CONFIG ────────────────────────────────────────────────────
BACKEND_URL   = "http://localhost:8000"
CHAT_ENDPOINT = f"{BACKEND_URL}/api/chat"

# Normal well-formed questions
EVAL_DATASET = [
    {
        "question": "What is neuromorphic computing?",
        "ground_truth": "Neuromorphic computing is a new class of hardware that imitates the way the human brain processes information through massively parallel, event-driven, and highly energy-efficient neural signals using spikes."
    },
    {
        "question": "What are spiking neural networks and how are they used?",
        "ground_truth": "Spiking neural networks process spikes over time for inference unlike traditional deep neural networks. They are used in neuromorphic hardware for energy-efficient event-driven computation."
    },
    {
        "question": "How much power does neuromorphic computing save?",
        "ground_truth": "Neuromorphic computing reduced power consumption by about 60 to 80 percent depending on input activity due to its event-driven calculation and infrequent spike processing."
    },
    {
        "question": "What is the difference between Von Neumann and neuromorphic architecture?",
        "ground_truth": "Von Neumann computers have separate CPUs and memory and are clock-driven. Neuromorphic systems combine processor and memory in neurons and synapses and are asynchronous and event-driven."
    },
    {
        "question": "What are the applications of neuromorphic computing for edge AI?",
        "ground_truth": "Applications include smart surveillance, autonomous robotics, wearable health monitoring, industrial IoT, smart home devices, automotive driver assistance, agricultural monitoring, and biomedical implants."
    },
    {
        "question": "What inference latency did the neuromorphic model achieve?",
        "ground_truth": "The neuromorphic model achieved sub-10ms inference latency consistently exceeding standard edge processors under real-time workloads."
    },
    {
        "question": "What chips are used in neuromorphic computing?",
        "ground_truth": "Neuromorphic chips include IBM TrueNorth, SpiNNaker, and Intel Loihi along with specialized hardware using memristors to implement spiking neural networks."
    },
    {
        "question": "What is STDP in neuromorphic computing?",
        "ground_truth": "STDP stands for Spike-Timing Dependent Plasticity, a local learning rule that allows neuromorphic models to adapt and learn at the edge without requiring cloud retraining."
    },
]

# Vague/poorly phrased versions of the same questions
# Used ONLY for Metric 5 (Query Rewrite Robustness)
VAGUE_DATASET = [
    {
        "vague":        "um what is that brain computer thing",
        "ground_truth": "Neuromorphic computing is a new class of hardware that imitates the way the human brain processes information through massively parallel, event-driven, and highly energy-efficient neural signals using spikes."
    },
    {
        "vague":        "how does the spike network work",
        "ground_truth": "Spiking neural networks process spikes over time for inference unlike traditional deep neural networks. They are used in neuromorphic hardware for energy-efficient event-driven computation."
    },
    {
        "vague":        "power savings something neuromorphic",
        "ground_truth": "Neuromorphic computing reduced power consumption by about 60 to 80 percent depending on input activity due to its event-driven calculation and infrequent spike processing."
    },
    {
        "vague":        "old computers vs new brain ones difference",
        "ground_truth": "Von Neumann computers have separate CPUs and memory and are clock-driven. Neuromorphic systems combine processor and memory in neurons and synapses and are asynchronous and event-driven."
    },
    {
        "vague":        "where can edge AI brain chips be used",
        "ground_truth": "Applications include smart surveillance, autonomous robotics, wearable health monitoring, industrial IoT, smart home devices, automotive driver assistance, agricultural monitoring, and biomedical implants."
    },
]

# ── AUTO INSTALL ──────────────────────────────────────────────
import subprocess, sys

PACKAGES = [
    "requests", "numpy",
    "bert-score",
    "sentence-transformers",
    "scikit-learn",
    "transformers",
    "torch",
]
for pkg in PACKAGES:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"📦 Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import re, json, time, requests
import numpy as np
from bert_score import score as bert_score_fn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def rewrite_query(q):
    try:
        res = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3:8b",
            "prompt": f"""Rewrite this vague question into a precise technical question.
Only return the rewritten question, nothing else.
Vague: {q}
Rewritten:""",
            "stream": False
        })
        return res.json()["response"].strip()
    except:
        return q

# ── QUERY BACKEND ─────────────────────────────────────────────
def query_backend(question):
    print("\nORIGINAL:", question)
    
    clean_question = question
    print("REWRITTEN:", clean_question)
    try:
        start = time.perf_counter()
        r = requests.post(
           CHAT_ENDPOINT,
           json={
            "message": clean_question,
            "api_key": "llama3"   # 🔥 IMPORTANT
        },
        timeout=120
        )
        ms = (time.perf_counter() - start) * 1000
        d = r.json()
        print("DEBUG RESPONSE:", d)
        answer = (d.get("response") or d.get("answer") or
                  d.get("message") or str(d))
        return {"answer": answer, "ms": ms, "error": None}
    except Exception as e:
        return {"answer": "", "ms": 0, "error": str(e)}
    
# ─────────────────────────────────────────────────────────────
print("\n" + "═"*62)
print("   AURA AI — 5-Metric RAG Evaluation Suite")
print("═"*62)

# ── STEP 1: Query Backend ─────────────────────────────────────
print("\n🔄 Querying backend for main questions...\n")
results = []
for item in EVAL_DATASET:
    out = query_backend(item["question"])
    results.append({**item, **out})
    icon = "✅" if not out["error"] else "❌"
    print(f"  {icon} {item['question'][:65]}")
    if not out["error"]:
        print(f"     → {out['answer'][:85]}...")
        print(f"     ⏱  {out['ms']:.0f} ms")
    else:
        print(f"     Error: {out['error']}")

valid = [r for r in results if not r["error"] and r["answer"]]
preds = [r["answer"]       for r in valid]
refs  = [r["ground_truth"] for r in valid]
print(f"\n  {len(valid)}/{len(results)} successful")

# ══════════════════════════════════════════════════════════════
# METRIC 1 — BERTScore F1
# ══════════════════════════════════════════════════════════════
print("\n" + "─"*62)
print("📊 METRIC 1 — BERTScore F1")
print("   Semantic correctness of answer vs ground truth")
print("─"*62)

_, _, F1 = bert_score_fn(preds, refs, lang="en", verbose=False)
bert_scores = F1.tolist()

for i, r in enumerate(valid):
    g = "🟢" if bert_scores[i] >= 0.75 else "🟡" if bert_scores[i] >= 0.55 else "🔴"
    print(f"  {g} [{bert_scores[i]:.4f}] {r['question'][:62]}")

mean_bert = float(np.mean(bert_scores))
g = "🟢 GOOD" if mean_bert >= 0.75 else "🟡 OK" if mean_bert >= 0.55 else "🔴 NEEDS IMPROVEMENT"
print(f"\n  ➤ Mean BERTScore F1 : {mean_bert:.4f}  {g}")

# ══════════════════════════════════════════════════════════════
# METRIC 2 — Answer Relevance
# ══════════════════════════════════════════════════════════════
print("\n" + "─"*62)
print("📊 METRIC 2 — Answer Relevance")
print("   Cosine similarity: Question ↔ Answer")
print("─"*62)

print("  Loading embedding model (all-MiniLM-L6-v2)...")
embed = SentenceTransformer("all-MiniLM-L6-v2")

relevance_scores = []
for r in valid:
    q_e = embed.encode([r["question"]])
    a_e = embed.encode([r["answer"]])
    sim = float(cosine_similarity(q_e, a_e)[0][0])
    relevance_scores.append(sim)
    g = "🟢" if sim >= 0.70 else "🟡" if sim >= 0.45 else "🔴"
    print(f"  {g} [{sim:.4f}] {r['question'][:62]}")

mean_rel = float(np.mean(relevance_scores))
g = "🟢 GOOD" if mean_rel >= 0.70 else "🟡 OK" if mean_rel >= 0.45 else "🔴 NEEDS IMPROVEMENT"
print(f"\n  ➤ Mean Answer Relevance : {mean_rel:.4f}  {g}")

# ══════════════════════════════════════════════════════════════
# METRIC 3 — Answer Correctness
# ══════════════════════════════════════════════════════════════
print("\n" + "─"*62)
print("📊 METRIC 3 — Answer Correctness")
print("   Cosine similarity: Answer ↔ Ground Truth")
print("─"*62)

correctness_scores = []
for r in valid:
    a_e  = embed.encode([r["answer"]])
    gt_e = embed.encode([r["ground_truth"]])
    sim  = float(cosine_similarity(a_e, gt_e)[0][0])
    correctness_scores.append(sim)
    g = "🟢" if sim >= 0.70 else "🟡" if sim >= 0.45 else "🔴"
    print(f"  {g} [{sim:.4f}] {r['question'][:62]}")

mean_corr = float(np.mean(correctness_scores))
g = "🟢 GOOD" if mean_corr >= 0.70 else "🟡 OK" if mean_corr >= 0.45 else "🔴 NEEDS IMPROVEMENT"
print(f"\n  ➤ Mean Answer Correctness : {mean_corr:.4f}  {g}")

# ══════════════════════════════════════════════════════════════
# METRIC 4 — Hallucination Score (NLI-based)
# ══════════════════════════════════════════════════════════════
print("\n" + "─"*62)
print("📊 METRIC 4 — Hallucination Score (NLI)")
print("   Detects if answer contradicts ground truth")
print("─"*62)

print("  Loading NLI model (cross-encoder/nli-MiniLM2-L6-H768)...")
try:
    from transformers import pipeline
    nli = pipeline(
        "text-classification",
        model="cross-encoder/nli-MiniLM2-L6-H768",
        device=-1  # CPU
    )

    hallucination_scores = []
    for r in valid:
        # NLI: premise=ground_truth, hypothesis=answer
        # ENTAILMENT → answer is faithful  → low hallucination
        # CONTRADICTION → answer contradicts → high hallucination
        inp = f"{r['ground_truth']} [SEP] {r['answer']}"
        out = nli(inp, truncation=True, max_length=512)
        label = out[0]["label"].upper()
        score = out[0]["score"]

        if label == "ENTAILMENT":
            hall_score = 1.0 - score      # low hallucination
        elif label == "CONTRADICTION":
            hall_score = score            # high hallucination
        else:  # NEUTRAL
            hall_score = 0.3

        hallucination_scores.append(hall_score)
        faithful = 1.0 - hall_score
        g = "🟢" if faithful >= 0.70 else "🟡" if faithful >= 0.45 else "🔴"
        print(f"  {g} [faithful={faithful:.4f}] {r['question'][:55]}  NLI={label}")

    mean_faithful = float(np.mean([1 - s for s in hallucination_scores]))
    mean_hall     = float(np.mean(hallucination_scores))
    g = "🟢 GOOD" if mean_faithful >= 0.70 else "🟡 OK" if mean_faithful >= 0.45 else "🔴 NEEDS IMPROVEMENT"
    print(f"\n  ➤ Mean Faithfulness Score : {mean_faithful:.4f}  {g}")
    print(f"  ➤ Mean Hallucination Risk  : {mean_hall:.4f}  {'🟢 LOW' if mean_hall <= 0.3 else '🟡 MEDIUM' if mean_hall <= 0.6 else '🔴 HIGH'}")
    nli_available = True

except Exception as e:
    print(f"  ⚠️  NLI model failed: {e}")
    print("  → Falling back to cosine contradiction estimate...")
    # Fallback: high cosine = faithful, low = possible hallucination
    hall_fallback = []
    for r in valid:
        a_e  = embed.encode([r["answer"]])
        gt_e = embed.encode([r["ground_truth"]])
        sim  = float(cosine_similarity(a_e, gt_e)[0][0])
        hall_fallback.append(sim)
    mean_faithful = float(np.mean(hall_fallback))
    mean_hall     = 1.0 - mean_faithful
    g = "🟢 GOOD" if mean_faithful >= 0.70 else "🟡 OK" if mean_faithful >= 0.45 else "🔴 NEEDS IMPROVEMENT"
    print(f"\n  ➤ Mean Faithfulness (fallback) : {mean_faithful:.4f}  {g}")
    nli_available = False

# ══════════════════════════════════════════════════════════════
# METRIC 5 — Query Rewrite Robustness
# ══════════════════════════════════════════════════════════════
print("\n" + "─"*62)
print("📊 METRIC 5 — Query Rewrite Robustness")
print("   Tests LangGraph agent on vague/poorly-phrased questions")
print("─"*62)

print("  Querying backend with vague questions...\n")
rewrite_scores = []

for item in VAGUE_DATASET:
    out = query_backend(item["vague"])
    if out["error"] or not out["answer"]:
        print(f"  ❌ Failed: {item['vague']}")
        continue

    # Compare vague answer vs ground truth using cosine
    a_e  = embed.encode([out["answer"]])
    gt_e = embed.encode([item["ground_truth"]])
    sim  = float(cosine_similarity(a_e, gt_e)[0][0])
    rewrite_scores.append(sim)

    g = "🟢" if sim >= 0.60 else "🟡" if sim >= 0.40 else "🔴"
    print(f"  {g} [{sim:.4f}] Vague: '{item['vague'][:50]}'")
    print(f"          Answer: {out['answer'][:75]}...")
    print(f"          ⏱  {out['ms']:.0f} ms\n")

mean_robust = float(np.mean(rewrite_scores)) if rewrite_scores else 0.0
g = "🟢 GOOD" if mean_robust >= 0.60 else "🟡 OK" if mean_robust >= 0.40 else "🔴 NEEDS IMPROVEMENT"
print(f"  ➤ Mean Rewrite Robustness : {mean_robust:.4f}  {g}")

# ══════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════
latencies = [r["ms"] for r in results if not r["error"]]

print("\n" + "═"*62)
print("   ✅ FINAL EVALUATION REPORT — AURA AI")
print("═"*62)

def badge(v, threshold_good=0.70, threshold_ok=0.45):
    if v is None: return "N/A"
    if v >= threshold_good: return "🟢 GOOD"
    if v >= threshold_ok:   return "🟡 OK"
    return "🔴 NEEDS IMPROVEMENT"

report_metrics = [
    ("1. BERTScore F1",           mean_bert,     0.75, 0.55),
    ("2. Answer Relevance",       mean_rel,      0.70, 0.45),
    ("3. Answer Correctness",     mean_corr,     0.70, 0.45),
    ("4. Faithfulness (NLI)",     mean_faithful, 0.70, 0.45),
    ("5. Rewrite Robustness",     mean_robust,   0.60, 0.40),
]

print()
for name, val, tg, tok in report_metrics:
    print(f"  {name:<28} : {val:.4f}   {badge(val, tg, tok)}")

print()
print(f"  {'Latency Mean':<28} : {np.mean(latencies):.0f} ms")
print(f"  {'Latency p95':<28} : {np.percentile(latencies, 95):.0f} ms")

# Overall score
overall = float(np.mean([mean_bert, mean_rel, mean_corr, mean_faithful, mean_robust]))
print(f"\n  {'── OVERALL RAG SCORE':<28} : {overall:.4f}   {badge(overall, 0.65, 0.45)}")

# Save JSON
report = {
    "BERTScore_F1":        round(mean_bert, 4),
    "Answer_Relevance":    round(mean_rel, 4),
    "Answer_Correctness":  round(mean_corr, 4),
    "Faithfulness_NLI":    round(mean_faithful, 4),
    "Rewrite_Robustness":  round(mean_robust, 4),
    "Overall_Score":       round(overall, 4),
    "Latency_Mean_ms":     round(float(np.mean(latencies)), 1),
    "Latency_p95_ms":      round(float(np.percentile(latencies, 95)), 1),
}
with open("rag_eval_5metrics_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n" + "═"*62)
print("💾 Saved → rag_eval_5metrics_report.json")
print("═"*62)