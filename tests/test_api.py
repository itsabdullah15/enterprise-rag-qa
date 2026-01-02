# tests/test_api.py
import pytest
import os
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

@pytest.mark.skipif(
    os.getenv('CI') == 'true', 
    reason="Skipping in CI because it requires Google Cloud credentials"
)
def test_ask_endpoint_returns_answer():
    """Test the /ask endpoint (only run locally with credentials)."""
    response = client.post(
        "/ask",
        json={"question": "What is stance classification?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data