# src/rag_pipeline.py
import json
import faiss
import lmstudio as lms
from .embedder import LMStudioEmbedder
from .retriever import RetrieverWithRerank
from .utils import VECTOR_DIR, logger


class RAGPipeline:
    def __init__(self, k=5):
        self.embedder = LMStudioEmbedder()
        self.index = faiss.read_index(f"{VECTOR_DIR}/docs.index")

        # Загружаем чанки
        self.docs = []
        with open(f"{VECTOR_DIR}/docs.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                self.docs.append(json.loads(line))

        self.k = k
        self.retriever = RetrieverWithRerank(self.index, self.docs)

    def search_docs(self, query):
        # Эмбеддинг вопроса
        query_vec = self.embedder.embed(query)
        top_docs = self.retriever.retrieve(query_vec=query_vec, k=self.k, top_n=50)
        return top_docs

    def ask(self, query):
        context_docs = self.search_docs(query)
        if not context_docs:
            return "❌ Нет подходящей информации в базе."

        context_texts = [d["text"] for d in context_docs]
        context = "\n\n".join(context_texts)
        sources = list(dict.fromkeys(d["source"] for d in context_docs))

        prompt = f"""
            Контекст:\n{context}
            Вопрос пользователя: {query}
            """

        logger.info("🤖 Генерация ответа ИИ...")
        lms.set_sync_api_timeout(600)
        model = lms.llm("lmstudio-community/mistral-7b-instruct")
        response_stream = model.respond_stream(prompt, config={"temperature": 0.0})

        answer = ""
        for fragment in response_stream:
            chunk = fragment.content
            print(chunk, end="", flush=True)
            answer += chunk

        print(f"\n📄 Источники:\n" + "\n".join(sources))
        return answer
