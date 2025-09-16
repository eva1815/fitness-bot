# embeddings_store.py
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

class EmbStore:
    def __init__(self, kb_items: List[Dict[str, str]], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.kb = kb_items[:]  # keep original order for stable indices
        self.model = SentenceTransformer(model_name)
        # We embed the KB "q" field (queries/prompts). You can also embed answers if you prefer.
        self.texts = [it["q"] for it in self.kb]
        self.embs = self.model.encode(self.texts, normalize_embeddings=True)
        self.nn = NearestNeighbors(n_neighbors=5, metric="cosine")
        self.nn.fit(self.embs)

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        q_emb = self.model.encode([query], normalize_embeddings=True)
        distances, idxs = self.nn.kneighbors(q_emb, n_neighbors=min(k, len(self.texts)))
        # cosine distance ∈ [0,2], similarity = 1 - distance
        results = []
        for d, i in zip(distances[0], idxs[0]):
            sim = 1.0 - float(d)
            item = self.kb[int(i)]
            results.append({
                "i": int(i),
                "score": sim,
                "q": item["q"],
                "a": item["a"]
            })
        return results
