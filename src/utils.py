# src/utils.py
import os
from pathlib import Path
from dotenv import load_dotenv
import logging

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = BASE_DIR / "vectorstore"

# chunking defaults
CHUNK_MAX_SIZE = int(os.getenv("CHUNK_MAX_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# LM Studio config
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL", "http://localhost:1234")
LM_STUDIO_EMBED_URL = os.getenv("LM_STUDIO_EMBED_URL", f"{LM_STUDIO_URL}/v1/embeddings")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "mistral-7b-instruct")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "")

for p in [DATA_DIR, VECTOR_DIR]:
    p.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag")
