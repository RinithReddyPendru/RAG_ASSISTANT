"""
Semantic L1 Cache & Graph Subgraph Acceleration Layer
Provides sub-10ms semantic query caching via cosine embedding similarity and
memoizes multi-hop entity neighborhoods for the Knowledge Graph.
"""

import time
import numpy as np
from typing import Optional, Dict, Any, List

class SemanticL1Cache:
    def __init__(self, similarity_threshold: float = 0.90, max_size: int = 500):
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.cache: List[Dict[str, Any]] = []
        self._graph_neighborhood_cache: Dict[str, List[str]] = {}
        self.stats = {
            "hits": 0,
            "misses": 0,
            "total_latency_saved_sec": 0.0
        }

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def lookup(self, query: str, query_embedding: List[float]) -> Optional[Dict[str, Any]]:
        """Searches cache for semantically matching queries above similarity threshold."""
        if not self.cache or not query_embedding:
            self.stats["misses"] += 1
            return None

        q_vec = np.array(query_embedding, dtype=np.float32)
        best_match = None
        best_sim = -1.0

        for entry in self.cache:
            sim = self._cosine_similarity(q_vec, entry["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_match = entry

        if best_sim >= self.similarity_threshold and best_match is not None:
            self.stats["hits"] += 1
            latency_saved = best_match.get("original_latency", 6.5)
            self.stats["total_latency_saved_sec"] += latency_saved
            result = dict(best_match["payload"])
            result["cache_hit"] = True
            result["cache_similarity"] = round(best_sim, 4)
            result["latency_saved"] = round(latency_saved, 2)
            return result

        self.stats["misses"] += 1
        return None

    def store(self, query: str, query_embedding: List[float], payload: Dict[str, Any], latency: float):
        """Stores a query embedding and its response payload in the semantic cache."""
        if not query_embedding:
            return

        if len(self.cache) >= self.max_size:
            self.cache.pop(0)  # Evict oldest

        self.cache.append({
            "query": query,
            "embedding": np.array(query_embedding, dtype=np.float32),
            "payload": payload,
            "original_latency": latency,
            "timestamp": time.time()
        })

    def get_cached_neighborhood(self, entity_key: str) -> Optional[List[str]]:
        return self._graph_neighborhood_cache.get(entity_key.lower())

    def store_cached_neighborhood(self, entity_key: str, triples: List[str]):
        if len(self._graph_neighborhood_cache) > 2000:
            self._graph_neighborhood_cache.clear()
        self._graph_neighborhood_cache[entity_key.lower()] = triples

    def clear(self):
        self.cache.clear()
        self._graph_neighborhood_cache.clear()

# Global singleton cache instance
semantic_cache = SemanticL1Cache()
