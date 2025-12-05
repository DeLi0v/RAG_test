# src/embedder.py
import numpy as np
import lmstudio as lms
from .utils import logger, EMBED_MODEL


class LMStudioEmbedder:
    """
    Uses LM Studio SDK for embeddings.
    Handles single strings or lists, batching, and returns normalized numpy arrays.
    """

    def __init__(self, model: str = None, batch_size: int = 16):
        self.model_name = model or EMBED_MODEL
        self.batch_size = batch_size
        logger.info("Using LM Studio embeddings with model=%s", self.model_name)
        # Получаем объект модели через SDK
        self.model = lms.embedding_model(self.model_name)

    def embed(self, texts):
        """
        texts: str or list[str]
        returns: numpy array (N, dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_emb = self.model.embed(
                batch
            )  # LM Studio SDK возвращает list[list[float]]
            embeddings.extend(batch_emb)

        embs_np = np.array(embeddings, dtype="float32")

        # нормализуем к единичной длине (косинусная схожесть через inner product)
        norms = np.linalg.norm(embs_np, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs_np = embs_np / norms

        return embs_np
