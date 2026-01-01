from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_ask_endpoint_returns_answer():
    response = client.post(
        "/ask",
        json={"question": "What is stance classification?"}
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert isinstance(data["answer"], str)
    assert len(data["answer"]) > 0