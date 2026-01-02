# tests/test_api.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

@patch('src.generation.llm.VertexGeminiLLM')
@patch('src.retrieval.vector_store.FAISSVectorStore')
def test_ask_endpoint_returns_answer(mock_faiss_class, mock_llm_class):
    """
    Test the /ask endpoint returns a valid answer.
    Mocks FAISSVectorStore and VertexGeminiLLM to avoid external dependencies.
    """
    # Mock FAISSVectorStore
    mock_faiss_instance = MagicMock()
    mock_faiss_instance.search.return_value = [
        {
            "score": 0.95,
            "metadata": {
                "text": "Stance classification is a task in NLP that involves determining the position or attitude expressed in text."
            }
        }
    ]
    mock_faiss_class.load.return_value = mock_faiss_instance
    
    # Mock VertexGeminiLLM instance
    mock_llm_instance = MagicMock()
    mock_llm_instance.generate.return_value = "Stance classification is a natural language processing task that identifies the position or attitude expressed in text toward a particular target or topic."
    mock_llm_class.return_value = mock_llm_instance
    
    # Make the request
    response = client.post(
        "/ask",
        json={"question": "What is stance classification?"}
    )
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
    assert "Stance classification" in data["answer"]
    
    # Verify mocks were called
    mock_faiss_class.load.assert_called_once()
    mock_faiss_instance.search.assert_called_once()
    mock_llm_class.assert_called_once()
    mock_llm_instance.generate.assert_called_once()