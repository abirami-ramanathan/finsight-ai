from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT)
)

from finsight.rag.indexer import FAISSIndexer

INDEX_DIR = ROOT / "data" / "rag_index"

indexer = FAISSIndexer.load(
    INDEX_DIR
)

queries = [
    "What is a mutual fund?",
    "How can an investor buy and sell shares?",
    "What are REITs?",
    "What are exchange traded funds?",
    "What are the rights of an investor?"
]

print("=" * 70)
print("MODULE D — RAG RETRIEVAL TEST")
print("=" * 70)

for query in queries:

    print("\n" + "-" * 70)
    print("QUERY:", query)
    print("-" * 70)

    results = indexer.search(
        query,
        top_k=3
    )

    for rank, result in enumerate(
        results,
        start=1
    ):
        print(
            f"\n[{rank}] "
            f"{result['source']}"
        )

        print(
            f"Distance: "
            f"{result['distance']:.4f}"
        )

        text = result["text"].replace(
            "\n",
            " "
        )

        print(
            text[:500]
        )