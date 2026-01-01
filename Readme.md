enterprise-rag-qa/
├── README.md                          # Project overview, setup, demo GIF/screenshots
├── .gitignore                         # Ignore __pycache__, .env, uploads/, vector_db/, etc.
├── requirements.txt                   # Frozen dependencies (pip freeze > requirements.txt)
├── pyproject.toml                     # Optional: for build tools like poetry (alternative to requirements.txt)
├── .env                               # Environment variables (e.g., OLLAMA_MODEL=qwen3:8b)
├── docker-compose.yml                 # Optional: for local deployment with Ollama container
├── Dockerfile                         # For containerizing the app
│
├── config/
│   └── config.yaml                    # Central config: chunk_size, overlap, top_k, embedding_model, etc.
│
├── data/
│   ├── raw/                           # Original uploaded documents (PDFs, DOCX, TXT) - gitignored
│   ├── processed/                     # Cleaned/chunked text files (optional, gitignored)
│   └── sample_documents/              # Few public-domain sample PDFs for demo (committed)
│       ├── company_policies.pdf
│       ├── technical_manual.pdf
│       └── research_paper_ai.pdf
│
├── vector_db/                         # FAISS index files (gitignored)
│   ├── faiss_index.index
│   └── metadata.pkl                   # Stores document metadata (source, page, etc.)
│
├── src/
│   ├── __init__.py
│   ├── main.py                        # Entry point: runs Streamlit app (streamlit run src/main.py)
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── loader.py                  # Handles PDF, DOCX, TXT loading (PyPDF2, docx, etc.)
│   │   ├── chunker.py                 # Text splitting strategies (recursive, semantic)
│   │   ├── embedder.py                # Generates embeddings (Sentence Transformers)
│   │   └── pipeline.py                # Full ingestion pipeline: load → chunk → embed → index
│   │
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── vector_store.py            # FAISS wrapper (save/load/index)
│   │   ├── hybrid_retriever.py        # Combines semantic (FAISS) + keyword (BM25)
│   │   └── reranker.py                # Optional cross-encoder reranking
│   │
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── llm.py                     # Ollama client wrapper (with prompts, streaming)
│   │   └── rag_chain.py               # Full RAG pipeline: retrieve → augment prompt → generate
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Retrieval: MRR, Recall@K | Generation: ROUGE, Ragas
│   │   ├── test_queries.json          # Ground-truth Q&A pairs for evaluation
│   │   └── evaluate.py                # Script to run full evaluation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                  # Configured logging
│       ├── helpers.py                 # Misc utilities (file paths, cleaning)
│       └── prompts.py                 # Prompt templates (system prompt, RAG prompt, etc.)
│
├── app/
│   ├── streamlit_app.py               # Streamlit UI: upload, index, chat interface
│   ├── pages/                         # Multi-page Streamlit (optional)
│   │   ├── 1_Upload_Documents.py
│   │   ├── 2_Chat.py
│   │   └── 3_Evaluation.py
│   └── assets/                        # Logos, CSS, images for UI
│       └── style.css
│
├── notebooks/                         # Jupyter notebooks for experimentation
│   ├── 01_data_exploration.ipynb
│   ├── 02_chunking_strategies.ipynb
│   ├── 03_embedding_comparison.ipynb
│   └── 04_rag_evaluation.ipynb
│
├── tests/                             # Unit and integration tests (pytest)
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   └── conftest.py
│
└── scripts/
    ├── ingest_documents.py            # CLI: python scripts/ingest_documents.py --folder data/raw
    ├── evaluate_rag.py                # CLI: python scripts/evaluate_rag.py
    └── rebuild_index.py               # Rebuild vector DB from scratch