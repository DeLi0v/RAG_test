import json
import numpy as np
import faiss
from src.embedder import LocalEmbedder
from src.lmstudio_client import get_lmstudio_client
import lmstudio as lms


class RAGPipeline:
    def __init__(self):
        self.embedder = LocalEmbedder()
        self.index = faiss.read_index("vectorstore/docs.index")

        # Загружаем чанки из jsonl
        self.docs = []
        with open("vectorstore/docs.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                self.docs.append(json.loads(line))

        self.client = get_lmstudio_client()

    def search(self, query, k=5):
        print("🔍 Поиск релевантной информации...")

        vec = self.embedder.embed(query)
        vec = np.array(vec, dtype="float32")
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)

        D, I = self.index.search(vec, k)
        results = [self.docs[int(i)] for i in I[0] if i != -1]

        print(f"✅ Информация найдена")
        print(f"✅ Найдено {len(results)} релевантных чанков")
        return results

    def ask(self, query):
        context_docs = self.search(query)
        context_texts = [d["text"] for d in context_docs]
        context = "\n\n".join(context_texts)
        sources = list(dict.fromkeys(d["source"] for d in context_docs))

        print("🤖 Генерация ответа ИИ...")

        prompt_system = f"""
        Контекст: \n{context}

        Вопрос пользователя: {query}
        """

        lms.set_sync_api_timeout(600)
        model = lms.llm("lmstudio-community/mistral-7b-instruct")

        respond_predicted = model.respond_stream(
            prompt_system,
            config={
                "temperature": 0.0,
            },
            on_prompt_processing_progress=(
                lambda progress: print(f"{round(progress*100)}% complete")
            ),
        )

        answer_text = ""
        for fragment in respond_predicted:
            chunk = fragment.content
            print(chunk, end="", flush=True)
            answer_text += chunk

        print(f"\n📄 Источники:\n" + "\n".join(sources))

        return ""
