# Enterprise-RAG-QA

Enterprise-RAG-QA is a production-ready Question-Answering system built using **Retrieval-Augmented Generation (RAG)**.  
It retrieves relevant information from enterprise documents (PDFs, reports, knowledge bases, etc.) and generates accurate answers using Large Language Models (LLMs).

This project follows real-world engineering standards:

- Modular RAG pipeline
- REST API using FastAPI
- Evaluation metrics framework
- Test suite with coverage
- CI regression protection
- Docker compatible

---

## 📂 Project Structure

```bash
enterprise-rag-qa/
│
├── app/                     # FastAPI application
│   └── main.py
│
├── src/
│   ├── evaluation/          # Evaluation + metrics
│   ├── generation/          # RAG pipeline + LLM orchestration
│   ├── ingestion/           # Chunking + embedding + indexing
│   ├── retrieval/           # Vector search components
│   ├── utils/               # Helper utilities
│   └── __init__.py
│
├── tests/                   # Unit + API tests
├── scripts/                 # Metric validation scripts
├── data/                    # Input document storage
├── vector_db/               # FAISS index files
├── .github/workflows/       # CI configuration
├── Dockerfile               # Build container image
├── requirements.txt         # Dependencies
└── README.md
```

## Features

• Retrieval-Augmented Generation (RAG) pipeline
• Chunk-based document ingestion
• Dense vector similarity search using FAISS
• LLM-powered answer synthesis
• Confidence scoring support
• Offline evaluation suite
• CI-protected quality gates
