from fastapi.testclient import TestClient
from app.main import app
from src.generation.rag_chain import RAGChain

client = TestClient(app)


def test_ask_endpoint_mocked(monkeypatch):

    def fake_run(self, question: str):
        return "mocked answer"

    monkeypatch.setattr(RAGChain, "run", fake_run)

    response = client.post(
        "/ask",
        json={"question": "What is stance classification?"}
    )

    assert response.status_code == 200
    data = response.json()

    assert data["answer"] == "mocked answer"