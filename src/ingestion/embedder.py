from typing import List
import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer



class SentenceTransformerEmbedder:
    """
    Wrapper around SentenceTransformers for embedding text chunks.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        """
        Generate embeddings for a list of LangChain Documents.
        """
        texts = [doc.page_content for doc in documents]
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True  # IMPORTANT for cosine similarity
        )
        return np.array(embeddings)