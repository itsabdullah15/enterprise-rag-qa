# tests/test_api.py
import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

@patch('src.generation.llm.VertexAI')
@patch('src.retrieval.vector_store.FAISSVectorStore.load')
def test_ask_endpoint_returns_answer(mock_faiss_load, mock_vertexai):
    # Create a mock FAISSVectorStore instance
    mock_faiss_instance = MagicMock()
    
    # Mock the search method
    mock_faiss_instance.search.return_value = [
        {
            "score": 0.95,
            "metadata": {
                "text": "Stance classification is a natural language processing task."
            }
        }
    ]
    mock_faiss_load.return_value = mock_faiss_instance
    
    # Mock Vertex AI
    mock_ai_response = Mock()
    mock_ai_response.text = "Stance classification is an NLP task that determines the position expressed in text."
    mock_vertexai_instance = Mock()
    mock_vertexai_instance.generate_content.return_value = mock_ai_response
    mock_vertexai.return_value = mock_vertexai_instance
    
    response = client.post(
        "/ask",
        json={"question": "What is stance classification?"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0