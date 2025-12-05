import lmstudio as lms
import numpy as np
from .utils import logger, EMBED_MODEL, LM_STUDIO_API_KEY


class LMStudioEmbedder:
    def __init__(self, model=None):
        self.model_name = model or EMBED_MODEL
        logger.info("Using LM Studio embeddings with model=%s", self.model_name)
        self.model = lms.embedding_model(self.model_name)

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.embed(texts)
        return np.array(embeddings, dtype="float32")
