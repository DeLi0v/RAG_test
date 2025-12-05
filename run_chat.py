# run_chat.py
from src.rag_query import RAGPipeline

rag = RAGPipeline()
print("Вопрос (выйти: q или й):")
while True:
    q = input("> ").strip()
    if not q:
        continue
    if q.lower() in ["q", "й", "exit", "выход"]:
        break
    print("\nОтвет:")
    ans = rag.ask(q)
    print("\n---\n")
