from src.rag_rerank import RAGRerankPipeline

rag = RAGRerankPipeline()

while True:
    q = input('\nВопрос (выход "q" или "й"): ').strip()
    if q.lower() in ["quit", "выйти", "q", "й"]:
        print("👋 Выход из программы.")
        break
    if not q:
        print("⚠️ Введите вопрос или команду для выхода.")
        continue
    print("\nОтвет:", rag.ask(q))
