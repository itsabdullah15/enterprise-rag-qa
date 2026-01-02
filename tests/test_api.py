# tests/test_api.py
import pytest
from unittest.mock import Mock, patch
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

@patch('src.generation.llm.VertexAI')
@patch('src.retrieval.vector_store.Chroma')
def test_ask_endpoint_returns_answer(mock_chroma, mock_vertexai):
    # Mock the Vertex AI response
    mock_ai_response = Mock()
    mock_ai_response.text = "This is a mocked answer about stance classification."
    mock_vertexai_instance = Mock()
    mock_vertexai_instance.generate_content.return_value = mock_ai_response
    mock_vertexai.return_value = mock_vertexai_instance
    
    # Mock ChromaDB response
    mock_chroma_instance = Mock()
    mock_chroma_instance.similarity_search.return_value = [
        Mock(page_content="Stance classification is a task in NLP...")
    ]
    mock_chroma.return_value = mock_chroma_instance
    
    # Make the request
    response = client.post(
        "/ask",
        json={"question": "What is stance classification?"}
    )
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["answer"] == "This is a mocked answer about stance classification."