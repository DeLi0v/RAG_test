# src/retriever.py
import numpy as np
from sentence_transformers import CrossEncoder
import json


class RetrieverWithRerank:
    """
    RAG Retriever + optional reranking with CrossEncoder.
    """

    def __init__(
        self,
        index,
        docs,
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_rerank=True,
        device="cpu",
    ):
        self.index = index
        self.docs = docs
        self.use_rerank = use_rerank
        self.device = device

        if use_rerank:
            self.reranker = CrossEncoder(cross_encoder_model, device=device)
        else:
            self.reranker = None

    def retrieve(self, query_vec, k=3, top_n=50):
        q = np.array(query_vec, dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)

        D, I = self.index.search(q, top_n)
        ids = [int(x) for x in I[0] if x != -1]
        candidates = [self.docs[i] for i in ids]

        if self.use_rerank and candidates:
            pairs = [
                (query_vec, c["text"]) for c in candidates
            ]  # <- query_text передается извне
            pairs = [(query_vec, doc["text"]) for doc in self.docs]
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            scored = list(zip(candidates, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            topk = [item[0] for item in scored[:k]]
            return topk
        else:
            return candidates[:k]
