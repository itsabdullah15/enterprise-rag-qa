from typing import List
import numpy as np
from langchain_core.documents import Document
from src.ingestion.embedder import SentenceTransformerEmbedder
from src.retrieval.vector_store import FAISSVectorStore

MIN_SCORE = 0.25

class Retriever:
    def __init__(self, top_k: int = 5, min_score: float = 0.25):
        self.embedder = SentenceTransformerEmbedder()
        self.store = FAISSVectorStore.load()
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, query: str) -> List[Document]:
        query_embedding = self.embedder.model.encode(
            [query], normalize_embeddings=True
        )

        results = self.store.search(query_embedding, top_k=self.top_k)

        documents = []
        for r in results:
            if r["score"] < self.min_score:
                continue  # 🔒 similarity gate

            doc = Document(
                page_content=r["metadata"]["text"],
                metadata={k: v for k, v in r["metadata"].items() if k != "text"}
            )
            doc.metadata["score"] = r["score"]
            documents.append(doc)

        return documents
