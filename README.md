# RAG_test — lokal RAG с Docling + LM Studio

1. Установите зависимости:
   python -m pip install -r requirements.txt

2. Запустите LM Studio локально и загрузите модели:
   - LLM: mistral-7b-instruct (или lmstudio-community/mistral-7b-instruct)
   - Embeddings: text-embedding-nomic-embed-text-v1.5

3. Настройте `.env` (опционально):
   LM_STUDIO_URL=http://localhost:1234
   LM_STUDIO_EMBED_URL=http://localhost:1234/v1/embeddings
   LM_STUDIO_MODEL=mistral-7b-instruct
   EMBED_MODEL=text-embedding-nomic-embed-text-v1.5

4. Положите документы в `data/` (.pdf, .docx, .xlsx, .txt).

5. Постройте индекс:
   python build_all.py

6. Запустите чат:
   python run_chat.py
