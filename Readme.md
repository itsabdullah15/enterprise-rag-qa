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

### Features

• Retrieval-Augmented Generation (RAG) pipeline
• Chunk-based document ingestion
• Dense vector similarity search using FAISS
• LLM-powered answer synthesis
• Confidence scoring support
• Offline evaluation suite
• CI-protected quality gates

### Answer Generation Pipeline

User Question
     ↓
Retriever (FAISS similarity search)
     ↓
Top-K relevant chunks
     ↓
RAGChain synthesizes response
     ↓
Final Answer

### 📊 Evaluation Metrics

✔ Recall@K     – whether relevant documents were retrieved
✔ MRR          – rank quality of retrieved documents
✔ Exact Match  – exact answer correctness
✔ Token-F1     – partial word-level similarity

These metrics are also used in CI to detect quality regression.


# 🏁 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/itsabdullah15/enterprise-rag-qa.git
cd enterprise-rag-qa
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # Mac / Linux
venv\Scripts\activate         # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment
```text
.env
```

## 📥 Ingesting Documents

Place your PDF files inside:

```text
data/raw_pdfs/
```

Run the ingestion pipeline:

```bash
python -m src.ingestion.loader
```

This will:

- Extract text  
- Chunk documents  
- Generate embeddings  
- Store vectors in FAISS  

---

## 🤖 API Server

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

Swagger UI:

```
http://127.0.0.1:8000/docs
```

---

## 📬 Example Request — POST `/ask`

```json
{
  "question": "What is stance classification?"
}
```

### 📭 Example Response

```json
{
  "answer": "Stance classification is the task of determining the expressed or implied opinion toward a target."
}
```

---

## 🧪 Running Tests

Run all tests:

```bash
pytest -vv
```

Run with coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

---

## 📊 Offline Evaluation

Run:

```bash
python -m src.evaluation.evaluate
```

Example Output:

```text
Recall@5:     0.92
MRR:          0.88
Exact Match:  0.64
Token-F1:     0.79
```

---

## 🔁 Continuous Integration (CI)

GitHub Actions automatically runs:

- Dependency installation  
- Unit tests  
- Coverage reporting  
- Evaluation metrics  
- Regression threshold checks  

This prevents silent accuracy degradation.

---

## 🐳 Docker Deployment (Optional)

Build the image:

```bash
docker build -t enterprise-rag-qa .
```

Run the container:

```bash
docker run -p 8000:8000 -v $(pwd)/.env:/app/.env enterprise-rag-qa
```

Service URL:

```
http://localhost:8000
```

---

## 🛠 Tech Stack

- Python  
- FastAPI  
- LangChain  
- Sentence-Transformers  
- FAISS  
- PyTest  
- GitHub Actions  
- Docker  

---

## 📌 Roadmap

- Streaming responses  
- Reranking support  
- Conversation memory  
- Feedback dashboard  
- Admin monitoring UI  
- Multi-tenant support  

---

## 🤝 Contributing

Pull Requests are welcome.  
Feel free to open issues and suggest improvements ✨