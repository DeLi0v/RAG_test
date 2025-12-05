from sentence_transformers import SentenceTransformer
import numpy as np

class LocalEmbedder:
    def __init__(self, model_name="nomic-ai/nomic-embed-text-v1.5"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, trust_remote_code=True)

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        emb = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return emb.tolist()
