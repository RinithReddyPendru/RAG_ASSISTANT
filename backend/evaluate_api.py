import json
import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevance,
    context_precision,
    context_recall,
)
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

API_URL = "http://localhost:8000/api/chat"

def is_relevant(retrieved_chunk: str, ground_truth_chunks: list) -> bool:
    """Check if the retrieved chunk contains the ground truth string."""
    for gt in ground_truth_chunks:
        if gt.lower() in retrieved_chunk.lower():
            return True
    return False

def calc_precision_at_k(retrieved_contexts: list, ground_truth_chunks: list, k: int) -> float:
    retrieved_k = retrieved_contexts[:k]
    if not retrieved_k:
        return 0.0
    relevant_count = sum(1 for chunk in retrieved_k if is_relevant(chunk, ground_truth_chunks))
    return relevant_count / k

def calc_mrr(retrieved_contexts: list, ground_truth_chunks: list) -> float:
    for rank, chunk in enumerate(retrieved_contexts, 1):
        if is_relevant(chunk, ground_truth_chunks):
            return 1.0 / rank
    return 0.0

def run_evaluation():
    dataset_path = os.path.join(os.path.dirname(__file__), "evaluation_dataset.json")
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return
        
    with open(dataset_path, "r") as f:
        eval_data = json.load(f)

    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    # IR Metric lists
    precisions_at_3 = []
    precisions_at_5 = []
    mrrs = []

    print(f"Sending {len(eval_data)} queries to the local API: {API_URL}")
    
    for idx, item in enumerate(eval_data):
        q = item['question']
        print(f"[{idx+1}/{len(eval_data)}] Querying: {q}")
        
        try:
            response = requests.post(
                API_URL, 
                json={"message": q, "api_key": "llama3.2", "include_context": True},
                timeout=180
            )
            response.raise_for_status()
            data = response.json()
            
            generated_answer = data.get("response", "")
            retrieved_docs = data.get("context", [])
            
        except Exception as e:
            print(f"API Error for question '{q}': {e}")
            generated_answer = ""
            retrieved_docs = []

        questions.append(q)
        answers.append(generated_answer)
        contexts.append(retrieved_docs)
        ground_truths.append([item.get('ground_truth', '')])
        
        # Calculate IR Metrics immediately
        gt_context = item.get('ground_truth_context', [])
        if gt_context:
            precisions_at_3.append(calc_precision_at_k(retrieved_docs, gt_context, 3))
            precisions_at_5.append(calc_precision_at_k(retrieved_docs, gt_context, 5))
            mrrs.append(calc_mrr(retrieved_docs, gt_context))
        else:
            precisions_at_3.append(0.0)
            precisions_at_5.append(0.0)
            mrrs.append(0.0)

    # 1. Output IR Metrics
    avg_p3 = sum(precisions_at_3)/len(precisions_at_3) if precisions_at_3 else 0
    avg_p5 = sum(precisions_at_5)/len(precisions_at_5) if precisions_at_5 else 0
    avg_mrr = sum(mrrs)/len(mrrs) if mrrs else 0
    
    print("\n--- Information Retrieval (IR) Metrics ---")
    print(f"Mean Reciprocal Rank (MRR): {avg_mrr:.2f}")
    print(f"Average Precision@3: {avg_p3:.2f}")
    print(f"Average Precision@5: {avg_p5:.2f}")
    
    # 2. Run RAGAS
    print("\nInitializing RAGAS evaluation (Local Ollama llama3.2)...")
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths
    })
    
    evaluator_llm = ChatOllama(model="llama3.2", temperature=0)
    evaluator_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    metrics = [faithfulness, answer_relevance, context_precision, context_recall]
    
    try:
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
        print("\n--- RAGAS Metrics ---")
        print(results)
        
        df = results.to_pandas()
        
        # Append IR metrics to DataFrame
        df['Precision@3'] = precisions_at_3
        df['Precision@5'] = precisions_at_5
        df['MRR'] = mrrs
        
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        report_path = os.path.join(log_dir, "eval_report_api.csv")
        df.to_csv(report_path, index=False)
        print(f"Evaluation exported to {report_path}")
        
        # 3. Generate Graphs
        print("Generating metric graphs...")
        plt.style.use('dark_background')
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # RAGAS metrics bar chart
        ragas_scores = {
            'Faithfulness': df['faithfulness'].mean(),
            'Answer Relevance': df['answer_relevance'].mean(),
            'Context Precision': df['context_precision'].mean(),
            'Context Recall': df['context_recall'].mean(),
        }
        
        axes[0].bar(ragas_scores.keys(), ragas_scores.values(), color=['#4caf50', '#2196f3', '#00bcd4', '#ff9800'])
        axes[0].set_ylim(0, 1.05)
        axes[0].set_title('Average RAGAS Generation Metrics')
        axes[0].set_ylabel('Score')
        axes[0].tick_params(axis='x', rotation=15)
        
        # Add labels on top of bars
        for idx, val in enumerate(ragas_scores.values()):
            axes[0].text(idx, val + 0.02, f'{val:.2f}', ha='center', fontweight='bold')
            
        # IR metrics bar chart
        ir_scores = {
            'MRR': avg_mrr,
            'Precision@3': avg_p3,
            'Precision@5': avg_p5
        }
        
        axes[1].bar(ir_scores.keys(), ir_scores.values(), color=['#e91e63', '#9c27b0', '#673ab7'])
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title('Average Information Retrieval Metrics')
        axes[1].set_ylabel('Score')
        
        # Add labels on top of bars
        for idx, val in enumerate(ir_scores.values()):
            axes[1].text(idx, val + 0.02, f'{val:.2f}', ha='center', fontweight='bold')
            
        plt.tight_layout()
        graph_path = os.path.join(log_dir, "evaluation_metrics_graph.png")
        plt.savefig(graph_path, dpi=300)
        print(f"Graphs saved to {graph_path}")
        
    except Exception as e:
        print(f"Error during RAGAS evaluation or graphing: {e}")

if __name__ == "__main__":
    run_evaluation()
