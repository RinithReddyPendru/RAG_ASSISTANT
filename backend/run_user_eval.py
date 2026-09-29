import requests
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

# Added these to utilize your local LLM instead of OpenAI keys
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------
# CONFIG
# -----------------------------
# Changed to your actual local endpoint
RAG_API = "http://localhost:8000/api/chat"   

# -----------------------------
# DATASET (FROM YOUR PAPER)
# -----------------------------
questions = [
    "What is neuromorphic computing?",
    "What is Edge AI?",
    "How is neuromorphic computing different from traditional computing?",
    "Why is neuromorphic computing energy efficient?",
    "What are spiking neural networks?",
    "What role do neuromorphic chips play in Edge AI?",
    "Give examples of applications of neuromorphic computing in Edge AI.",
    "Why is neuromorphic computing suitable for real-time applications?",
    "What advantages does neuromorphic computing provide over traditional edge processors?",
    "How does event-driven processing improve efficiency in neuromorphic systems?"
]

ground_truths = [
    "Brain-inspired computing using spikes and parallel processing",
    "Running AI on edge devices without cloud",
    "Parallel event-driven vs sequential clock-driven",
    "Computes only on spikes reducing power",
    "Neural networks using spikes over time",
    "Enable low power real-time processing",
    "Surveillance robotics wearable IoT smart homes",
    "Parallel asynchronous low latency",
    "Low power real-time noise handling local learning",
    "Only computes when needed saving energy"
]

# -----------------------------
# FETCH FROM YOUR RAG BACKEND
# -----------------------------
answers = []
contexts = []

print("\nFetching responses from RAG backend...\n")

for q in questions:
    try:
        # Adapted JSON payload to match ChatRequest format
        response = requests.post(RAG_API, json={"message": q, "api_key": "llama3.2", "include_context": True})
        
        if response.status_code != 200:
            print(f"Server returned {response.status_code}: {response.text}")
            answers.append("")
            contexts.append([""])
            continue
            
        data = response.json()

        answer = data.get("response", "")
        # Changed "contexts" to "context" to match the modified app.py
        context = data.get("context", [""]) 
        
        # Ensure context is at least a list even if empty
        if not context:
            context = [""]

        answers.append(answer)
        contexts.append(context)

        print(f"Q: {q}")
        print(f"A: {answer}\n")

    except Exception as e:
        print("Error:", e)
        answers.append("")
        contexts.append([""])

# -----------------------------
# CREATE DATASET FOR RAGAS
# -----------------------------
# Note: Ragas requires ground_truths to be a list of lists of strings
ground_truths_list = [[gt] for gt in ground_truths]

dataset = Dataset.from_dict({
    "user_input": questions,
    "response": answers,
    "retrieved_contexts": contexts,
    "reference": ground_truths
})

# -----------------------------
# RUN RAGAS METRICS
# -----------------------------
print("\nRunning RAGAS evaluation with local llama3.2...\n")

evaluator_llm = ChatOllama(model="llama3.2", temperature=0)
evaluator_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

try:
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )

    print("\nRAGAS RESULTS:\n", result)

    # -----------------------------
    # PLOT GRAPH (Only run if RAGAS succeeded)
    # -----------------------------
    metrics_names = ["Faithfulness", "Relevancy", "Precision", "Recall", "MRR"]
    values = [
        result.get("faithfulness", 0),
        result.get("answer_relevancy", 0),
        result.get("context_precision", 0),
        result.get("context_recall", 0)
    ]
except Exception as e:
    print(f"RAGAS evaluation failed: {e}")
    # Fallback if evaluation fails
    values = [0, 0, 0, 0]
    metrics_names = ["Faithfulness", "Relevancy", "Precision", "Recall", "MRR"]

# -----------------------------
# MRR METRIC
# -----------------------------
def mrr(contexts):
    scores = []
    for ctx in contexts:
        rank = 0
        for i, c in enumerate(ctx):
            if len(c) > 20:  # simple relevance condition
                rank = i + 1
                break
        scores.append(1 / rank if rank else 0)
    return sum(scores) / len(scores) if scores else 0

mrr_score = mrr(contexts)

print("\nMRR:", mrr_score)

# Append MRR to values for plotting
values.append(mrr_score)

# -----------------------------
# SAVE RESULTS (CSV)
# -----------------------------
df = pd.DataFrame({
    "question": questions,
    "answer": answers
})

df.to_csv("evaluation_results.csv", index=False)
print("\nSaved results to evaluation_results.csv")

# -----------------------------
# PLOT GRAPH
# -----------------------------
plt.figure()
plt.bar(metrics_names, values, color=['blue', 'orange', 'green', 'red', 'purple'])
plt.title("RAG Evaluation Metrics")
plt.xlabel("Metrics")
plt.ylabel("Score")
plt.savefig("evaluation_graph.png")
# plt.show() # Commented out GUI plot so it doesn't hang terminal execution

print("\nGraph saved as evaluation_graph.png")
