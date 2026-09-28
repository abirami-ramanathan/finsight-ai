from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []

    for document in documents:
        parts = splitter.split_text(document["text"])

        for index, part in enumerate(parts):
            chunks.append({
                "source": document["source"],
                "chunk_id": index,
                "text": part
            })

    return chunks