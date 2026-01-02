# tests/test_api.py
import pytest
from unittest.mock import Mock, patch, MagicMock
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

@patch('vertexai.generative_models.GenerativeModel')
@patch('src.retrieval.vector_store.FAISSVectorStore')
def test_ask_endpoint_returns_answer(mock_faiss_class, mock_gen_model):
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
    
    # Mock GenerativeModel
    mock_model_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Stance classification is a natural language processing task that identifies the position or attitude expressed in text toward a particular target or topic."
    mock_model_instance.generate_content.return_value = mock_response
    mock_gen_model.return_value = mock_model_instance
    
    # Also mock vertexai.init to prevent initialization
    with patch('vertexai.init'):
        response = client.post(
            "/ask",
            json={"question": "What is stance classification?"}
        )
    
    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0
    
    # Verify mocks were called
    mock_faiss_class.load.assert_called_once()
    mock_faiss_instance.search.assert_called_once()

## Alternative: Mock the VertexGeminiLLM class directly


# tests/test_api.py
# import pytest
# from unittest.mock import Mock, patch, MagicMock
# from app.main import app
# from fastapi.testclient import TestClient

# client = TestClient(app)

# @patch('src.generation.llm.VertexGeminiLLM')
# @patch('src.retrieval.vector_store.FAISSVectorStore')
# def test_ask_endpoint_returns_answer(mock_faiss_class, mock_llm_class):
#     # Mock FAISSVectorStore
#     mock_faiss_instance = MagicMock()
#     mock_faiss_instance.search.return_value = [
#         {
#             "score": 0.95,
#             "metadata": {
#                 "text": "Stance classification is a task in NLP."
#             }
#         }
#     ]
#     mock_faiss_class.load.return_value = mock_faiss_instance
    
#     # Mock VertexGeminiLLM instance
#     mock_llm_instance = MagicMock()
#     mock_llm_instance.generate.return_value = "Stance classification is an NLP task that determines position in text."
#     mock_llm_class.return_value = mock_llm_instance
    
#     response = client.post(
#         "/ask",
#         json={"question": "What is stance classification?"}
#     )
    
#     assert response.status_code == 200
#     data = response.json()
#     assert "answer" in data
#     assert len(data["answer"]) > 0