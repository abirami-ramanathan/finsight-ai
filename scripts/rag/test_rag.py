from pathlib import Path
import sys


ROOT = Path(
    __file__
).resolve().parents[2]


sys.path.insert(
    0,
    str(ROOT)
)


from finsight.rag.service import RAGService


service = RAGService(
    index_dir=ROOT / "data" / "rag_index",
    top_k=5,
    root_dir=ROOT
)


questions = [
    "What is a mutual fund?",
    "What is a REIT?",
    "What is an ETF?",
    "How can an investor buy and sell shares?",
    "What technical indicators does FinSight AI use?",
    "What forecasting results does FinSight AI have for ITC?",
    "What forecasting model performed well for BTC?",
    "How does FinSight AI use forecasting results for recommendations?",
    "How does LinUCB improve the recommendations?",
    "What were the results of the 2000 LinUCB evaluation?"
]


print("=" * 70)
print("MODULE D — EXPLAINABLE FINANCIAL RAG")
print("=" * 70)


for question in questions:

    result = service.ask(
        question
    )

    print("\n" + "=" * 70)

    print("QUESTION:")
    print(question)

    print("\nROUTED TO:")
    print(result["module"])

    print("\nANSWER:")
    print(result["answer_with_citations"])