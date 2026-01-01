from typing import List
from collections import defaultdict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken

# Initialize tokenizer ONCE (performance-safe)
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _token_length(text: str) -> int:
    """Return token length for text."""
    return len(_TOKENIZER.encode(text))


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 512,
    overlap: int = 50
) -> List[Document]:
    """
    Split documents into token-aware chunks while preserving metadata.
    Chunk IDs are assigned PER DOCUMENT.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=_token_length,
        separators=["\n\n", "\n", " ", ""],  # semantic-first splitting
    )

    chunks = splitter.split_documents(documents)

    # Assign chunk_id per document (enterprise-correct)
    doc_chunk_counter = defaultdict(int)

    for chunk in chunks:
        doc_id = chunk.metadata["document_id"]
        chunk.metadata["chunk_id"] = doc_chunk_counter[doc_id]
        doc_chunk_counter[doc_id] += 1

    print(
        f"✂️ Chunked {len(documents)} pages into {len(chunks)} chunks "
        f"({chunk_size} tokens, overlap={overlap})"
    )

    return chunks