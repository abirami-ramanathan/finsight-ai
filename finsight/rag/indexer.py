from pathlib import Path
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class FAISSIndexer:
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []

    def build(self, chunks):
        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        ).astype("float32")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexHNSWFlat(
            dimension,
            32
        )

        self.index.hnsw.efConstruction = 40
        self.index.hnsw.efSearch = 32

        self.index.add(embeddings)

        self.metadata = chunks

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            str(directory / "index.faiss")
        )

        with open(
            directory / "metadata.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                self.metadata,
                f,
                ensure_ascii=False,
                indent=2
            )

    @classmethod
    def load(
        cls,
        directory,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    ):
        directory = Path(directory)

        obj = cls(model_name)

        obj.index = faiss.read_index(
            str(directory / "index.faiss")
        )

        with open(
            directory / "metadata.json",
            encoding="utf-8"
        ) as f:
            obj.metadata = json.load(f)

        return obj

    def search(self, query, top_k=5):
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        distances, indices = self.index.search(
            embedding,
            top_k
        )

        results = []

        for distance, index in zip(
            distances[0],
            indices[0]
        ):
            if index < 0:
                continue

            item = self.metadata[int(index)].copy()
            item["distance"] = float(distance)
            results.append(item)

        return results