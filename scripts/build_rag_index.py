import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DOCUMENTS_DIR, MODEL_DIR, EMBEDDING_MODEL
from finsight.rag.indexer import read_documents, RAGIndex

docs = read_documents(DOCUMENTS_DIR)
if not docs:
    raise SystemExit("Put PDF/TXT/MD financial documents into data/documents first.")

idx = RAGIndex(EMBEDDING_MODEL).build(docs)
idx.save(MODEL_DIR / "rag")
print("Indexed", len(idx.chunks), "chunks")
