import json
import os
import pandas as pd
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

# Import existing application graph from RAG Assistant
from rag_agent import agentic_rag

def run_evaluation():
    print("Loading test dataset...")
    # 1. Load Ground Truth Data
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

    print(f"Found {len(eval_data)} test cases. Running inference through LangGraph pipeline...")
    
    # 2. Run inference for all test queries
    for idx, item in enumerate(eval_data):
        print(f"Processing question {idx + 1}/{len(eval_data)}: {item['question']}")
        
        # Invoke agent
        state = {
            "question": item['question'],
            "model_name": "llama3.2",
            "loop_step": 0
        }
        
        result = agentic_rag.invoke(state)
        
        # Extract generation and context documents
        generated_answer = result.get("generation", "Failed to generate answer")
        retrieved_docs = result.get("documents", [])
        context_texts = [doc.page_content for doc in retrieved_docs]
        
        questions.append(item['question'])
        answers.append(generated_answer)
        contexts.append(context_texts)
        ground_truths.append([item.get('ground_truth', '')]) # Ragas expects ground truth as list of strings

    # 3. Create Dataset for Ragas
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths
    }
    
    dataset = Dataset.from_dict(data_dict)
    
    print("\nInference complete. Initializing local Ollama judges for RAGAS evaluation...")
    print("WARNING: Evaluating locally with Ollama takes a significant amount of time. Please be patient...")
    
    # Initialize local evaluator LLMs
    evaluator_llm = ChatOllama(model="llama3.2", temperature=0)
    evaluator_embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Measure Metrics
    metrics = [
        faithfulness,
        answer_relevance,
        context_precision,
        context_recall,
    ]
    
    try:
        # Run Ragas evaluation
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
        
        print("\nEvaluation Results:", results)
        
        # 5. Export to CSV for research paper
        df = results.to_pandas()
        
        # Make logs directory if it doesn't exist
        log_dir = os.path.join(os.path.dirname(__file__), "..", "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        report_path = os.path.join(log_dir, "eval_report.csv")
        df.to_csv(report_path, index=False)
        print(f"\nEvaluation successfully exported to {report_path}")
        
    except Exception as e:
        print(f"Error during RAGAS evaluation: {e}")

if __name__ == "__main__":
    run_evaluation()
