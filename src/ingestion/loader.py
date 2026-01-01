from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

RAW_PDF_BASE = Path("data/raw_pdfs")


def load_all_pdfs(base_path: Path = RAW_PDF_BASE) -> List[Document]:
    """
    Load all PDFs recursively from data/raw_pdfs/*
    Supports:
      - arxiv
      - hr_docs
      - any future folders
    """
    documents: List[Document] = []

    for pdf_path in base_path.rglob("*.pdf"):
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        document_id = f"{pdf_path.parent.name}/{pdf_path.name}"

        for doc in docs:
            doc.metadata.update({
                "document_id": document_id,           # UNIQUE doc identifier
                "source_type": pdf_path.parent.name,  # arxiv | hr_docs
                "source_file": pdf_path.name,
                "full_path": str(pdf_path),
                # page is already added by PyPDFLoader
            })

        documents.extend(docs)

        print(f"📄 Loaded {pdf_path.name} ({len(docs)} pages)")

    print(f"\n✅ Total pages loaded: {len(documents)}")
    return documents


if __name__ == "__main__":
    load_all_pdfs(RAW_PDF_BASE)