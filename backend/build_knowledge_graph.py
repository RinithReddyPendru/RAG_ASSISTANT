"""
Knowledge Graph Construction Utility for Aura AI RAG
Extracts key entities, components, and relationships across document domains:
- Aura AI & RAG Architecture
- Dental Insurance Systems & Algorithmic Claims
- Neuromorphic Computing & Spiking Neural Networks
- Maternal Health & Pregnancy Diet Models
Persists into data/graph.pkl (NetworkxEntityGraph) for fast, robust graph retrieval.
"""

import os
import pickle
import networkx as nx

GRAPH_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "graph.pkl")
os.makedirs(os.path.dirname(GRAPH_DIR), exist_ok=True)

class MockEntityGraph:
    """Compatible wrapper for NetworkxEntityGraph."""
    def __init__(self, nx_graph=None):
        self._graph = nx_graph if nx_graph is not None else nx.MultiDiGraph()
        
    def add_node(self, node_id, **kwargs):
        self._graph.add_node(node_id, **kwargs)
        
    def add_edge(self, source, target, **kwargs):
        self._graph.add_edge(source, target, **kwargs)

def build_domain_knowledge_graph():
    print("Building rich multi-domain Knowledge Graph...")
    G = nx.MultiDiGraph()

    # Domain 1: Aura AI & Adaptive RAG Architecture
    domain1_triples = [
        ("Aura AI", "implements", "Adaptive RAG Architecture"),
        ("Adaptive RAG Architecture", "uses", "LangGraph State Machine"),
        ("LangGraph State Machine", "orchestrates", "Adaptive Strategy Router"),
        ("Adaptive Strategy Router", "classifies", "Query Intent Space"),
        ("Adaptive Strategy Router", "selects", "Direct Retrieval Strategy"),
        ("Adaptive Strategy Router", "selects", "Vector Semantic Strategy"),
        ("Adaptive Strategy Router", "selects", "Knowledge Graph Strategy"),
        ("Adaptive Strategy Router", "selects", "DW-RRF Hybrid Strategy"),
        ("Vector Semantic Strategy", "queries", "ChromaDB"),
        ("ChromaDB", "stores", "Dense Semantic Embeddings"),
        ("Dense Semantic Embeddings", "computed_by", "all-MiniLM-L6-v2"),
        ("Knowledge Graph Strategy", "traverses", "NetworkX Graph Triples"),
        ("DW-RRF Hybrid Strategy", "calibrates", "Entity Density Index (EDI)"),
        ("DW-RRF Hybrid Strategy", "applies", "Reciprocal Rank Fusion"),
        ("Reciprocal Rank Fusion", "balances", "Lambda Vector Weight"),
        ("Reciprocal Rank Fusion", "balances", "Lambda Graph Weight"),
        ("DW-RRF Hybrid Strategy", "feeds", "Neural Cross-Encoder (FlashRank)"),
        ("Neural Cross-Encoder (FlashRank)", "prunes", "Lexical Distractors"),
        ("Neural Cross-Encoder (FlashRank)", "evaluates", "Context Sufficiency Scorer"),
        ("Context Sufficiency Scorer", "triggers", "Self-Correcting Refinement Loop"),
        ("Self-Correcting Refinement Loop", "reformulates", "Ambiguous Query Intent"),
        ("Context Sufficiency Scorer", "routes_to", "Grounded Response Generator"),
        ("Grounded Response Generator", "executed_by", "Ollama Cloud (120B Model)")
    ]

    # Domain 2: Indian Dental Insurance Systems & Automated Processing
    domain2_triples = [
        ("Indian Dental Insurance", "exhibits", "Six Major Structural Gaps"),
        ("Six Major Structural Gaps", "includes", "Rigid Non-Personalized Policies"),
        ("Six Major Structural Gaps", "includes", "Lack of Preventive AMC Subscriptions"),
        ("Six Major Structural Gaps", "includes", "Lengthy Waiting Periods"),
        ("Six Major Structural Gaps", "includes", "Slow Manual Claim Settlement"),
        ("Six Major Structural Gaps", "includes", "Uniform Exclusionary Pricing"),
        ("Six Major Structural Gaps", "includes", "Inadequate Senior Citizen Coverage"),
        ("Slow Manual Claim Settlement", "causes", "High Claim Rejection Rates"),
        ("High Claim Rejection Rates", "resolved_by", "AI-Driven Hybrid Claim Processing"),
        ("AI-Driven Hybrid Claim Processing", "provides", "Instant Cashless Payouts"),
        ("AI-Driven Hybrid Claim Processing", "integrates", "Empanelled Dental Clinics"),
        ("Empanelled Dental Clinics", "submits", "Digital Treatment Evidence"),
        ("Uniform Exclusionary Pricing", "mitigated_by", "Tiered Rural Micro-Subsidies"),
        ("Tiered Rural Micro-Subsidies", "supports", "Low-Income Subscribers"),
        ("Preventive AMC Subscriptions", "reduces", "Advanced Periodontal Surgery Costs"),
        ("Lack of Preventive AMC Subscriptions", "addressed_by", "Annual Maintenance Contracts")
    ]

    # Domain 3: Neuromorphic Computing for Edge AI
    domain3_triples = [
        ("Neuromorphic Computing", "utilizes", "Spiking Neural Networks (SNN)"),
        ("Spiking Neural Networks (SNN)", "transmits", "Discrete Event Spikes"),
        ("Discrete Event Spikes", "enables", "Event-Driven Processing"),
        ("Event-Driven Processing", "drastically_lowers", "Edge Power Consumption"),
        ("Neuromorphic Computing", "implements", "Synaptic Plasticity (STDP)"),
        ("Synaptic Plasticity (STDP)", "updates", "On-Chip Synaptic Weights"),
        ("Neuromorphic Hardware", "features", "Crossbar Memristor Arrays"),
        ("Crossbar Memristor Arrays", "accelerates", "In-Memory Matrix Multiplication"),
        ("Edge AI Hardware", "deploys", "Neuromorphic Cores"),
        ("Neuromorphic Cores", "processes", "Real-Time Sensor Streams"),
        ("Real-Time Sensor Streams", "connects_to", "Low-Latency Edge Diagnostics")
    ]

    # Domain 4: Maternal Healthcare & Pregnancy Diet Models
    domain4_triples = [
        ("Maternal Healthcare Model", "recommends", "Trimester-Specific Dietary Guidelines"),
        ("Trimester-Specific Dietary Guidelines", "specifies", "First Trimester Micronutrients"),
        ("First Trimester Micronutrients", "mandates", "Folic Acid Supplementation"),
        ("Folic Acid Supplementation", "prevents", "Neural Tube Defects"),
        ("Trimester-Specific Dietary Guidelines", "specifies", "Second Trimester Caloric Increase"),
        ("Second Trimester Caloric Increase", "requires", "High Protein & Calcium Intake"),
        ("Trimester-Specific Dietary Guidelines", "specifies", "Third Trimester Iron Density"),
        ("Third Trimester Iron Density", "prevents", "Maternal Anemia & Preterm Labor"),
        ("Rural Maternal Healthcare", "coordinated_by", "ASHA Healthcare Workers"),
        ("ASHA Healthcare Workers", "interfaces_with", "Primary Health Centres (PHC)"),
        ("Primary Health Centres (PHC)", "monitors", "Gestational Complication Risks"),
        ("Gestational Complication Risks", "includes", "Gestational Diabetes & Preeclampsia")
    ]

    # Inter-domain Cross-links (Enables Multi-hop Graph Traversal!)
    cross_domain_triples = [
        ("Aura AI", "indexes_domain", "Indian Dental Insurance"),
        ("Aura AI", "indexes_domain", "Neuromorphic Computing"),
        ("Aura AI", "indexes_domain", "Maternal Healthcare Model"),
        ("Edge AI Hardware", "runs", "AI-Driven Hybrid Claim Processing"),
        ("Primary Health Centres (PHC)", "benefits_from", "Tiered Rural Micro-Subsidies"),
        ("Low-Latency Edge Diagnostics", "deployed_in", "Primary Health Centres (PHC)")
    ]

    all_triples = domain1_triples + domain2_triples + domain3_triples + domain4_triples + cross_domain_triples

    for src, rel, dst in all_triples:
        G.add_node(src, type="Entity")
        G.add_node(dst, type="Entity")
        G.add_edge(src, dst, relation=rel)

    with open(GRAPH_DIR, "wb") as f:
        pickle.dump(G, f)

    print(f"[OK] Knowledge Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print(f"[SAVED] Persisted to: {GRAPH_DIR}")
    return G

if __name__ == "__main__":
    build_domain_knowledge_graph()
