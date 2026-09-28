from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(ROOT)
)

from finsight.rag.document_loader import load_documents
from finsight.rag.chunker import chunk_documents
from finsight.rag.indexer import FAISSIndexer

DOCUMENT_DIR = ROOT / "data" / "documents"
INDEX_DIR = ROOT / "data" / "rag_index"

documents = load_documents(
    DOCUMENT_DIR
)

print("=" * 70)
print("MODULE D — RAG KNOWLEDGE BASE")
print("=" * 70)

print(
    f"Documents loaded: {len(documents)}"
)

if not documents:
    print(
        "\nNo documents found."
    )
    print(
        "Place PDF, TXT, or MD financial documents in:"
    )
    print(DOCUMENT_DIR)
    raise SystemExit(1)

chunks = chunk_documents(
    documents
)

print(
    f"Chunks created: {len(chunks)}"
)

indexer = FAISSIndexer()

indexer.build(
    chunks
)

indexer.save(
    INDEX_DIR
)

print(
    f"FAISS index saved to: {INDEX_DIR}"
)

print("\nModule D Step 1 completed.")