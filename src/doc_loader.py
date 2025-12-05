# src/doc_loader.py
from pathlib import Path
from docling.document_converter import DocumentConverter
import logging

logger = logging.getLogger("rag")


def load_with_docling(path: str):
    """
    Returns list of chunks: {"text": ..., "source": ..., "type": "table"|"text"}
    """
    converter = DocumentConverter()
    doc = converter.convert(Path(path))  # ConversionResult
    result = doc.document.export_to_markdown()
    return result


def load_txt_file(path: str) -> str:
    encodings = ["utf-8", "cp1251", "latin-1"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
            print(f"[TXT] decoded with {enc}: {path}")
            return text
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError(f"Cannot decode TXT file: {path}")


def chunk_text(text: str, max_size: int, overlap: int):
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + max_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += max_size - overlap  # скользящее окно
        if start < 0:  # safety
            start = 0
        if start >= text_length:
            break
    return chunks
