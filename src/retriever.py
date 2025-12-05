# src/retriever.py
import numpy as np
from sentence_transformers import CrossEncoder
import json
import os
from typing import List, Dict, Any, Optional, Tuple


def load_docs(jsonl_path: str = "vectorstore/docs.jsonl") -> List[Dict[str, Any]]:
    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def faiss_search(
    index, query_vec: np.ndarray, top_n: int = 50
) -> Tuple[List[float], List[int]]:
    """
    Возвращает (distances, indices) для query_vec.
    query_vec должен иметь форму (1, dim) или (dim,)
    """
    q = np.array(query_vec, dtype="float32")
    if q.ndim == 1:
        q = q.reshape(1, -1)
    D, I = index.search(q, top_n)
    return D[0].tolist(), I[0].tolist()


class RetrieverWithRerank:
    def __init__(
        self,
        index,
        docs: List[Dict[str, Any]],
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_rerank: bool = True,
        device: str = "cpu",
    ):
        """
        index: faiss index instance
        docs: list of {"text":..., "source":...}
        """
        self.index = index
        self.docs = docs
        self.use_rerank = use_rerank
        self.device = device

        if use_rerank:
            # CrossEncoder — модель для reranking (query, passage) -> score
            self.reranker = CrossEncoder(cross_encoder_model, device=device)
        else:
            self.reranker = None

    def retrieve(
        self,
        query_text: str,
        query_vec: np.ndarray,
        k: int = 3,
        top_n: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Возвращает список top-k объектов docs (каждый — dict с полями text, source).
        Параметры:
        - query_text: текст запроса (строка), нужен для reranker
        - query_vec: вектор запроса (numpy array), нужен для FAISS поиска
        - top_n: сколько кандидатов брать с FAISS для последующего rerank
        - k: сколько вернуть в итог
        """
        # 1) FAISS поиск
        D, I = faiss_search(self.index, query_vec, top_n=top_n)
        ids = [int(x) for x in I if x != -1]
        candidates = [self.docs[i] for i in ids]

        if self.use_rerank and self.reranker is not None and len(candidates) > 0:
            # 2) подготовка пар (query_text, passage_text) для CrossEncoder
            pairs = [(query_text, c["text"]) for c in candidates]
            # 3) получение скорингов
            scores = self.reranker.predict(pairs, show_progress_bar=False)
            # 4) объединение кандидатов со скором и сортировка по убыванию
            scored = list(zip(candidates, scores))
            scored.sort(key=lambda x: x[1], reverse=True)
            topk = [item[0] for item in scored[:k]]
            return topk
        else:
            # fallback: просто возвращаем первые k кандидатов от FAISS
            return candidates[:k]
