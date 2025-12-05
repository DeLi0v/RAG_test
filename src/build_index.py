import os
import faiss
import numpy as np
from src.embedder import LocalEmbedder
from pathlib import Path
import logging
from docx import Document
import PyPDF2
import pandas as pd

logger = logging.getLogger("rag")

# Максимальный размер чанка (символов)
MAX_CHUNK_SIZE = 1000
# Шаг перекрытия между чанками
CHUNK_OVERLAP = 200

def load_documents(folder_path):
    """Загрузка файлов и разбиение на чанки"""
    docs = []
    folder = Path(folder_path)

    for file in folder.iterdir():
        if not file.is_file():
            continue

        text = ""
        try:
            if file.suffix.lower() == ".txt":
                try:
                    with open(file, "r", encoding="utf-8") as f:
                        text = f.read()
                except UnicodeDecodeError:
                    with open(file, "r", encoding="cp1251") as f:
                        text = f.read()
            elif file.suffix.lower() == ".docx":
                doc = Document(file)
                text = "\n".join([p.text for p in doc.paragraphs])
            elif file.suffix.lower() == ".pdf":
                with open(file, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
            elif file.suffix.lower() in [".xls", ".xlsx"]:
                xls = pd.ExcelFile(file)
                for sheet_name in xls.sheet_names:
                    df = pd.read_excel(xls, sheet_name=sheet_name, dtype=str)
                    text += "\n".join(df.fillna("").astype(str).agg(" ".join, axis=1)) + "\n"
            else:
                logger.warning(f"Формат не поддерживается: {file.name}")
                continue

            # Разбиваем на чанки
            start = 0
            text = text.strip()
            while start < len(text):
                end = start + MAX_CHUNK_SIZE
                chunk_text = text[start:end]
                docs.append({"text": chunk_text, "source": str(file.name)})
                start += MAX_CHUNK_SIZE - CHUNK_OVERLAP

        except Exception as e:
            logger.error(f"Ошибка при чтении {file.name}: {e}")

    logger.info(f"Загружено и разбито на {len(docs)} чанков")
    return docs

def build_faiss_index(docs, embedder):
    vectors = embedder.embed([d["text"] for d in docs])
    vectors_np = np.array(vectors).astype("float32")

    dim = vectors_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors_np)

    return index, vectors_np

def save_index(index, vectors_np, docs, out_dir="vectorstore"):
    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, f"{out_dir}/docs.index")
    np.save(f"{out_dir}/docs.npy", vectors_np)

    with open(f"{out_dir}/docs.jsonl", "w", encoding="utf-8") as f:
        import json
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

def build_index():
    embedder = LocalEmbedder()
    docs = load_documents("data/")
    index, vectors_np = build_faiss_index(docs, embedder)
    save_index(index, vectors_np, docs)
    print("FAISS index built successfully!")

if __name__ == "__main__":
    build_index()
