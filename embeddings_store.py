# embeddings_store.py  (Render-friendly, no torch)
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class EmbStore:
    def __init__(self, kb_items: List[Dict[str, str]]):
        """
        kb_items: list of {"q": "...", "a": "..."}
        """
        self.kb = kb_items[:]
        self.texts = [it["q"] for it in self.kb]
        # simple English stop-words; tiny and memory-friendly
        self.vect = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
        self.matrix = self.vect.fit_transform(self.texts)

    def search(self, query: str, k: int = 3) -> List[Dict]:
        if not query.strip():
            return []
        qv = self.vect.transform([query])
        sims = cosine_similarity(qv, self.matrix).ravel()
        # top-k indices by similarity
        top_idx = sims.argsort()[::-1][:k]
        hits = []
        for i in top_idx:
            hits.append({
                "i": i,
                "q": self.kb[i]["q"],
                "a": self.kb[i]["a"],
                "score": float(sims[i]),
            })
        return hits
