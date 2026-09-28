from pathlib import Path
from pypdf import PdfReader

def load_documents(directory):
    directory = Path(directory)
    documents = []

    for path in directory.rglob("*"):
        if path.suffix.lower() == ".pdf":
            reader = PdfReader(str(path))
            text = "\n".join(
                page.extract_text() or ""
                for page in reader.pages
            )

            if text.strip():
                documents.append({
                    "source": path.name,
                    "text": text
                })

        elif path.suffix.lower() in [".txt", ".md"]:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore"
            )

            if text.strip():
                documents.append({
                    "source": path.name,
                    "text": text
                })

    return documents