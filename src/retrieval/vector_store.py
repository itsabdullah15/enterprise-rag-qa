import faiss
import pickle
from pathlib import Path
from typing import List
import numpy as np
from langchain_core.documents import Document


VECTOR_DB_DIR = Path("vector_db")
VECTOR_DB_DIR.mkdir(exist_ok=True)


class FAISSVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner Product (cosine with normalized vectors)
        self.metadata: List[dict] = []

    def add(self, embeddings: np.ndarray, documents: List[Document]):
        """
        Add embeddings + metadata to FAISS index.
        """
        self.index.add(embeddings)

        for doc in documents:
            self.metadata.append({
                **doc.metadata,
                "text": doc.page_content
            })

    def save(self):
        """
        Persist FAISS index and metadata to disk.
        """
        faiss.write_index(self.index, str(VECTOR_DB_DIR / "faiss.index"))

        with open(VECTOR_DB_DIR / "metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

        print("💾 FAISS index and metadata saved")

    @classmethod
    def load(cls):
        """
        Load FAISS index and metadata from disk.
        """
        index = faiss.read_index(str(VECTOR_DB_DIR / "faiss.index"))

        with open(VECTOR_DB_DIR / "metadata.pkl", "rb") as f:
            metadata = pickle.load(f)

        store = cls(index.d)
        store.index = index
        store.metadata = metadata

        print("📦 FAISS index loaded")
        return store

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Perform similarity search.
        """
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "score": float(score),
                "metadata": self.metadata[idx]
            })

        return results