# src/rag_rerank.py
import json
import numpy as np
import faiss
from tqdm import tqdm
from src.embedder import LocalEmbedder
from src.lmstudio_client import get_lmstudio_client
from src.retriever import load_docs, RetrieverWithRerank
import lmstudio as lms


class RAGRerankPipeline:
    def __init__(
        self,
        index_path="vectorstore/docs.index",
        docs_path="vectorstore/docs.jsonl",
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="cpu",
    ):
        self.embedder = LocalEmbedder()
        self.index = faiss.read_index(index_path)
        self.docs = load_docs(docs_path)
        self.client = get_lmstudio_client()
        # используем reranker по умолчанию
        self.retriever = RetrieverWithRerank(
            self.index,
            self.docs,
            cross_encoder_model=cross_encoder_model,
            use_rerank=True,
            device=device,
        )

    def search_and_rerank(self, query, top_n=50, k=3):
        query_vec = np.array(self.embedder.embed(query), dtype="float32")
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        top_chunks = self.retriever.retrieve(
            query_text=query, query_vec=query_vec, k=3, top_n=50
        )

        return top_chunks

    def ask(self, query, model_name="mistral-7b-instruct", temperature=0.0):
        # 1. retrieve top-k high-quality chunks
        top_chunks = self.search_and_rerank(query, top_n=50, k=3)
        if not top_chunks:
            return "По данному запросу не найден релевантный контекст."

        # 2. build context and unique sources
        context_texts = [c["text"] for c in top_chunks]
        context = "\n\n".join(context_texts)
        sources = list(dict.fromkeys(c["source"] for c in top_chunks))

        # 3. build prompt (user role; avoid system role for LM Studio models that don't support it)
        user_message = f"""
                        Контекст:
                        {context}

                        Вопрос: {query}
                        """

        # 4. call LM Studio (stream)
        lms.set_sync_api_timeout(
            600
        )  # отключаем ограничение ожидания (или large value)
        model = lms.llm(model_name)

        print("🤖 Генерация ответа ИИ... (streaming)")
        answer_text = ""
        prediction_stream = model.respond_stream(
            user_message,
            config={"temperature": temperature},
            on_prompt_processing_progress=(
                lambda progress: print(f"{round(progress*100)}% complete")
            ),
        )

        # iterate fragments — model.respond_stream yields fragments (check your SDK version — often iterable)
        for frag in prediction_stream:
            chunk = getattr(frag, "content", "")  # frag.content
            print(chunk, end="", flush=True)
            answer_text += chunk

        print("\n✅ Ответ готов\n")
        print("📄 Источники:")
        for s in sources:
            print("-", s)

        return answer_text
