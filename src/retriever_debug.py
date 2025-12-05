# src/retriever_debug.py
import numpy as np
import json
from sentence_transformers import CrossEncoder
import faiss


def load_docs(jsonl_path="vectorstore/docs.jsonl"):
    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs


def debug_search(
    index_path="vectorstore/docs.index",
    embeddings_path="vectorstore/embeddings.npy",
    query_text="test",
    top_n=20,
    rerank_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cpu",
):
    index = faiss.read_index(index_path)
    docs = load_docs()

    # embed query using same embedder as index — здесь вставьте ваш embedder
    from src.embedder import LocalEmbedder

    embedder = LocalEmbedder()
    qvec = np.array(embedder.embed(query_text), dtype="float32")
    # normalize
    if qvec.ndim == 1:
        qvec = qvec.reshape(1, -1)
    qvec = qvec / (np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-9)

    D, I = index.search(qvec, top_n)
    print("FAISS candidates (score=inner product ~ cosine):")
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        doc = docs[int(idx)]
        print(
            f"idx={idx} score={score:.4f} source={doc.get('source')} type={doc.get('type')}"
        )
        print("  preview:", doc["text"][:200].replace("\n", " ").strip())
        print("---")

    # rerank using CrossEncoder
    pairs = [(query_text, docs[int(idx)]["text"]) for idx in I[0] if idx != -1]
    reranker = CrossEncoder(rerank_model, device=device)
    scores = reranker.predict(pairs, show_progress_bar=True)
    print("\nRerank scores (desc):")
    scored = [
        (int(idx), s) for idx, s in zip([idx for idx in I[0] if idx != -1], scores)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    for idx, s in scored[:10]:
        doc = docs[int(idx)]
        print(
            f"idx={idx} rerank={s:.4f} source={doc.get('source')} type={doc.get('type')}"
        )
        print("  preview:", doc["text"][:200].replace("\n", " ").strip())
        print("===")
    return scored
