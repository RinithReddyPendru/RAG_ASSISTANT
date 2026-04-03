from typing_extensions import TypedDict
from typing import List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from rag_core import get_vector_store, get_graph

class GraphState(TypedDict):
    """
    Represents the state of our graph.
    """
    question: str
    generation: str
    documents: List[Document]
    model_name: str
    loop_step: int

def retrieve(state: GraphState):
    """Retrieve documents using Hybrid Vector + Graph search."""
    print("---RETRIEVE---")
    question = state["question"]
    model_name = state["model_name"]
    
    documents = []
    
    # 1. Standard Vector Retrieval (Maximized accuracy: k=5)
    retriever = get_vector_store().as_retriever(search_kwargs={"k": 5})
    vector_docs = retriever.invoke(question)
    documents.extend(vector_docs)
    
    # 2. Knowledge Graph Retrieval
    graph = get_graph()
    if hasattr(graph, '_graph'):
        try:
            # Simple keyword extraction, highly capped for speed
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Extract 1-3 key entity nouns from the question. Output them as a comma separated list ONLY. DO NOT EXPLAIN."),
                ("human", "{question}")
            ])
            llm = ChatOllama(model=model_name, temperature=0, num_predict=15)
            entities_str = (prompt | llm | StrOutputParser()).invoke({"question": question})
            
            entities = [e.strip() for e in entities_str.split(",") if e.strip()]
            graph_context = []
            
            nx_graph = graph._graph
            # Traverse graph for matching entities
            for ent in entities:
                for node in nx_graph.nodes():
                    if ent.lower() in str(node).lower():
                        for neighbor in nx_graph.neighbors(node):
                            edge_data = nx_graph.get_edge_data(node, neighbor)
                            relation = edge_data.get("relation", "is related to")
                            graph_context.append(f"{node} --{relation}--> {neighbor}")
                            
            if graph_context:
                graph_doc = Document(page_content="Knowledge Graph Internal Relations:\n" + "\n".join(set(graph_context)))
                documents.append(graph_doc)
                print("---ADDED GRAPH CONTEXT---")
        except Exception as e:
            print(f"Graph retrieval error (non-fatal): {e}")

    return {"documents": documents, "question": question}

def grade_documents(state: GraphState):
    """Determines whether the retrieved context contains relevant information (Optimized to Single Pass)."""
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION (HARDWARE SPEED BYPASS)---")
    # To prevent 2-5 minute execution times on local hardware, we completely bypass 
    # the LLM grader. It will now instantly pass all retrieved docs to the generator.
    return {"documents": state["documents"], "question": state["question"]}

def generate(state: GraphState):
    """Generate answer."""
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    model_name = state["model_name"]
    
    if not documents:
        return {"generation": "I am sorry, but there was not enough context in the documents to answer this safely."}
    
    llm = ChatOllama(model=model_name, temperature=0.2)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an intelligent information assistant. Use the following context to answer the question.\n\nKeep your answer extremely brief and concise, maximum 1-3 sentences! Do not ramble.\n\nContext:\n{context}"),
        ("human", "{question}"),
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = prompt | llm | StrOutputParser()
    
    generation = rag_chain.invoke({"context": format_docs(documents), "question": question})
    return {"generation": generation, "documents": documents, "question": question}

def rewrite(state: GraphState):
    """Transform the query to produce a better question."""
    print("---REWRITE QUESTION---")
    question = state["question"]
    model_name = state["model_name"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a question re-writer that converts an input question to a better version optimized for vectorstore retrieval. \n"
         "Look at the input and try to reason about the underlying semantic intent. Output ONLY your rewritten question."),
        ("human", "Initial question: {question} \n Formulate an improved question."),
    ])
    
    llm = ChatOllama(model=model_name, temperature=0)
    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": question})
    
    return {"question": better_question, "loop_step": state.get("loop_step", 0) + 1}

def route_question(state: GraphState):
    """Route depending on document relevance."""
    filtered_docs = state["documents"]
    loop_step = state.get("loop_step", 0)
    
    if not filtered_docs:
        if loop_step >= 2:
            print("---DECISION: MAX RETRIES REACHED, FORCING GENERATE---")
            return "generate"
        print("---DECISION: ALL DOCUMENTS ARE IRRELEVANT, REWRITE QUESTION---")
        return "rewrite"
    else:
        print("---DECISION: DOCUMENTS ARE RELEVANT, GENERATE---")
        return "generate"

# Build Graph
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("rewrite", rewrite)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", route_question)
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

agentic_rag = workflow.compile()
