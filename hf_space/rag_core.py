import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import NetworkxEntityGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
import pickle

DB_DIR = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "docs")
GRAPH_DIR = os.path.join(os.path.dirname(__file__), "data", "graph.pkl")

# Initialize embeddings locally (free)
try:
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception as e:
    print(f"Warning: Could not load embeddings immediately, might download on first use. Error: {e}")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vector_store():
    # Load existing or create new Chroma DB
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

import networkx as nx

class KnowledgeGraphWrapper:
    """Standard wrapper around NetworkX MultiDiGraph for entity-relation retrieval."""
    def __init__(self, nx_graph=None):
        self._graph = nx_graph if nx_graph is not None else nx.MultiDiGraph()
        
    def add_node(self, node_id, **kwargs):
        self._graph.add_node(node_id, **kwargs)
        
    def add_edge(self, source, target, **kwargs):
        self._graph.add_edge(source, target, **kwargs)

def get_graph():
    if os.path.exists(GRAPH_DIR):
        try:
            with open(GRAPH_DIR, "rb") as f:
                obj = pickle.load(f)
                if hasattr(obj, "_graph"):
                    return obj
                return KnowledgeGraphWrapper(obj)
        except Exception as e:
            print(f"Warning loading graph.pkl: {e}")
    return KnowledgeGraphWrapper()

def save_graph(graph):
    with open(GRAPH_DIR, "wb") as f:
        # Save underlying networkx graph
        nx_g = getattr(graph, "_graph", graph)
        pickle.dump(nx_g, f)

def ingest_documents(model_name: str = "llama3", file_path: str = None):
    """Loads docs and ingests them into Chroma."""
    print(f"DEBUG: ingest_documents called for {file_path}")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    loaders = []
    if file_path and os.path.exists(file_path):
        if file_path.endswith('.pdf'):
            loaders.append(PyPDFLoader(file_path))
        else:
            loaders.append(TextLoader(file_path))
    else:
        loaders = [
            DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader),
            DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
        ]
    
    docs = []
    for loader in loaders:
        try:
            print(f"DEBUG: loading {loader}")
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {loader}: {e}")
            
    if not docs:
        print("No documents found to ingest.")
        return 0
        
    print(f"DEBUG: splitting {len(docs)} docs")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    
    # Store directly to chroma
    print(f"DEBUG: embedding and storing {len(splits)} splits into Chroma")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_DIR)
    print("DEBUG: chroma done")
    
    # Graph Extraction
    try:
        print("Extracting Graph Entities (this may take a bit for local models)...")
        llm = ChatOllama(model=model_name, temperature=0, base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
        llm_transformer = LLMGraphTransformer(llm=llm)
        
        print("DEBUG: Graph conversion starting...")
        # Process a subset to save time if large
        graph_documents = llm_transformer.convert_to_graph_documents(splits[:1])
        print("DEBUG: Graph conversion finished.")
        
        graph = get_graph()
        for g_doc in graph_documents:
            for node in g_doc.nodes:
                graph.add_node(node.id, type=node.type)
            for rel in g_doc.relationships:
                graph.add_edge(rel.source.id, rel.target.id, relation=rel.type)
                
        save_graph(graph)
        print("Knowledge Graph saved.")
    except Exception as e:
        print(f"Graph extraction failed or skipped: {e}")
        
    return len(splits)

def get_rag_chain(api_key: str):
    """Returns a RAG chain configured. For Ollama, the api_key field is actually the model name."""
    model_name = api_key if api_key else "llama3"
        
    # Using local Ollama model
    llm = ChatOllama(model=model_name, temperature=0.2, base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    vectorstore = get_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    system_prompt = (
        "You are an intelligent information assistant. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use detailed yet concise responses and format nicely with markdown.\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
