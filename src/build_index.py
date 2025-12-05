# src/build_index.py
import os
import json
from pathlib import Path
import numpy as np
import faiss
from tqdm import tqdm
from .doc_loader import load_with_docling, load_txt_file
from .embedder import LMStudioEmbedder
from .utils import VECTOR_DIR, DATA_DIR, logger, CHUNK_MAX_SIZE, CHUNK_OVERLAP


def chunk_text(text, max_size=CHUNK_MAX_SIZE, overlap=CHUNK_OVERLAP):
    text = text.strip()
    if len(text) <= max_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_size
        chunks.append(text[start:end].strip())
        start += max_size - overlap
    return chunks


def load_all_and_chunk(data_folder):
    docs = []
    folder = Path(data_folder)

    for file in folder.iterdir():
        if file.is_dir():
            docs.extend(load_all_and_chunk(str(file)))
            continue

        try:
            if file.suffix.lower() == ".txt":
                text = load_txt_file(str(file))
            else:
                text = load_with_docling(str(file))

            if not text.strip():
                continue

            chunks = chunk_text(text, CHUNK_MAX_SIZE, CHUNK_OVERLAP)
            for chunk in chunks:
                docs.append(
                    {
                        "text": chunk,
                        "source": file.name,
                        "type": "text",  # TODO: можно добавить распознавание таблиц
                    }
                )
            logger.info("Parsed %s -> %d chunks", file.name, len(chunks))
        except Exception as e:
            logger.exception("Error parsing %s: %s", file.name, e)

    logger.info("Total chunks: %d", len(docs))
    return docs


def build_index(data_folder=DATA_DIR, out_dir=VECTOR_DIR, batch_size=16):
    os.makedirs(out_dir, exist_ok=True)
    embedder = LMStudioEmbedder(batch_size=batch_size)
    docs = load_all_and_chunk(data_folder)

    if not docs:
        logger.error("No texts found to index")
        return

    texts = [d["text"] for d in docs]
    logger.info("Generating embeddings for %d chunks", len(texts))

    # Батчинг с tqdm для прогресс-бара
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks", ncols=80):
        batch = texts[i : i + batch_size]
        batch_emb = embedder.embed(batch)
        embeddings.append(batch_emb)

    embs_np = np.vstack(embeddings)

    # создаем FAISS индекс с косинусной схожестью (inner product)
    dim = embs_np.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs_np)

    # сохраняем индекс и метаданные
    faiss.write_index(index, str(Path(out_dir) / "docs.index"))
    np.save(Path(out_dir) / "embeddings.npy", embs_np)

    with open(Path(out_dir) / "docs.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    logger.info("Index built: %s (dim=%d, n=%d)", out_dir, dim, len(texts))


if __name__ == "__main__":
    build_index()
