"""
Tests for the chat API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from app.main import app
from app.vector_store import VectorStore
from app.chat_workflow import ChatRAGWorkflow


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_vector_store():
    mock_store = Mock(spec=VectorStore)
    mock_store.documents = ["doc1", "doc2", "doc3"]
    return mock_store


@pytest.fixture
def mock_chat_workflow():
    mock_workflow = Mock(spec=ChatRAGWorkflow)
    mock_workflow.process_chat_query.return_value = {
        "user_id": "test_user",
        "query": "test query", 
        "response": "This is a test response",
        "conversation_context": {"is_followup_question": False},
        "search_enhancement": {"optimized_queries": ["test"]},
        "metadata": {
            "retrieval": {"top_k": 3, "total_results": 2},
            "generation": {"model": "claude-3-haiku-20240307", "documents_used": 2},
            "documents_used": 2,
            "has_error": False
        },
        "debug_info": None
    }
    return mock_workflow


class TestChatAPI:
    
    def test_root_endpoint_serves_web_app(self, client):
        """Test that root endpoint serves the chat web app."""
        response = client.get("/")
        assert response.status_code == 200
        # Should serve HTML content
        assert "text/html" in response.headers.get("content-type", "")
    
    def test_api_info_endpoint(self, client):
        """Test API info endpoint shows chat-only endpoints."""
        response = client.get("/api")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Product Chat Bot API"
        assert data["version"] == "2.0.0"
        assert "chat" in data["endpoints"]
        assert "conversation" in data["endpoints"]
        assert "health" in data["endpoints"]
        # Should NOT have query endpoint
        assert "query" not in data["endpoints"]
    
    def test_health_endpoint_healthy(self, client, mock_vector_store):
        """Test health endpoint with healthy vector store."""
        # Set the global vector_store variable directly instead of patching
        import app.main
        original_vector_store = app.main.vector_store
        app.main.vector_store = mock_vector_store
        
        try:
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
            assert data["version"] == "2.0.0"
            assert data["vector_store_status"] == "healthy"
            assert data["indexed_documents"] == 3
        finally:
            app.main.vector_store = original_vector_store
    
    def test_health_endpoint_empty_store(self, client):
        """Test health endpoint with empty vector store."""
        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.documents = []
        
        import app.main
        original_vector_store = app.main.vector_store
        app.main.vector_store = mock_vector_store
        
        try:
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
            assert data["vector_store_status"] == "empty"
            assert data["indexed_documents"] == 0
        finally:
            app.main.vector_store = original_vector_store
    
    def test_chat_endpoint_success(self, client, mock_chat_workflow):
        """Test successful chat query."""
        import app.main
        original_workflow = app.main.chat_rag_workflow
        
        # Configure the mock to return the actual query passed in
        def mock_process_chat_query(user_id, query):
            return {
                "user_id": user_id,
                "query": query,
                "response": "This is a test response",
                "conversation_context": {"is_followup_question": False},
                "search_enhancement": {"optimized_queries": ["test"]},
                "metadata": {
                    "retrieval": {"top_k": 3, "total_results": 2},
                    "generation": {"model": "claude-3-haiku-20240307", "documents_used": 2},
                    "documents_used": 2,
                    "has_error": False
                },
                "debug_info": None
            }
        
        mock_chat_workflow.process_chat_query.side_effect = mock_process_chat_query
        app.main.chat_rag_workflow = mock_chat_workflow
        
        try:
            request_data = {
                "user_id": "test_user",
                "query": "What smartphones do you have?"
            }
            
            response = client.post("/chat", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["user_id"] == "test_user"
            assert data["query"] == "What smartphones do you have?"
            assert data["response"] == "This is a test response"
            assert "conversation_context" in data
            assert "search_enhancement" in data
            assert data["metadata"]["has_error"] is False
            
            mock_chat_workflow.process_chat_query.assert_called_once_with(
                "test_user", "What smartphones do you have?"
            )
        finally:
            app.main.chat_rag_workflow = original_workflow
    
    def test_chat_endpoint_with_error(self, client):
        """Test chat endpoint when workflow returns error."""
        mock_workflow = Mock(spec=ChatRAGWorkflow)
        mock_workflow.process_chat_query.return_value = {
            "user_id": "test_user",
            "query": "test query",
            "response": "Error occurred",
            "conversation_context": {"is_followup_question": False},
            "search_enhancement": {"optimized_queries": []},
            "metadata": {
                "retrieval": {"error": True},
                "generation": {"error": True},
                "documents_used": 0,
                "has_error": True
            },
            "debug_info": {"error": "Something went wrong"}
        }
        
        import app.main
        original_workflow = app.main.chat_rag_workflow
        app.main.chat_rag_workflow = mock_workflow
        
        try:
            request_data = {
                "user_id": "test_user",
                "query": "test query"
            }
            
            response = client.post("/chat", json=request_data)
            assert response.status_code == 200  # API still returns 200, error is in response
            
            data = response.json()
            assert data["metadata"]["has_error"] is True
            assert data["debug_info"]["error"] == "Something went wrong"
        finally:
            app.main.chat_rag_workflow = original_workflow
    
    def test_chat_endpoint_validation_error_empty_user_id(self, client):
        """Test chat endpoint validation with empty user ID."""
        # Set workflow to avoid dependency issues during validation
        import app.main
        original_workflow = app.main.chat_rag_workflow
        app.main.chat_rag_workflow = Mock()
        
        try:
            request_data = {
                "user_id": "",
                "query": "test query"
            }
            
            response = client.post("/chat", json=request_data)
            assert response.status_code == 422  # Validation error
        finally:
            app.main.chat_rag_workflow = original_workflow
    
    def test_chat_endpoint_validation_error_empty_query(self, client):
        """Test chat endpoint validation with empty query."""
        import app.main
        original_workflow = app.main.chat_rag_workflow
        app.main.chat_rag_workflow = Mock()
        
        try:
            request_data = {
                "user_id": "test_user",
                "query": ""
            }
            
            response = client.post("/chat", json=request_data)
            assert response.status_code == 422  # Validation error
        finally:
            app.main.chat_rag_workflow = original_workflow
    
    def test_chat_endpoint_validation_error_missing_fields(self, client):
        """Test chat endpoint validation with missing fields."""
        import app.main
        original_workflow = app.main.chat_rag_workflow
        app.main.chat_rag_workflow = Mock()
        
        try:
            request_data = {
                "user_id": "test_user"
                # Missing query field
            }
            
            response = client.post("/chat", json=request_data)
            assert response.status_code == 422  # Validation error
        finally:
            app.main.chat_rag_workflow = original_workflow
    
    def test_chat_endpoint_internal_server_error(self, client):
        """Test chat endpoint with internal server error."""
        mock_workflow = Mock(spec=ChatRAGWorkflow)
        mock_workflow.process_chat_query.side_effect = Exception("Unexpected error")
        
        import app.main
        original_workflow = app.main.chat_rag_workflow
        app.main.chat_rag_workflow = mock_workflow
        
        try:
            request_data = {
                "user_id": "test_user",
                "query": "test query"
            }
            
            response = client.post("/chat", json=request_data)
            assert response.status_code == 500
            
            data = response.json()
            assert "Internal server error" in data["detail"]
        finally:
            app.main.chat_rag_workflow = original_workflow
    
    def test_get_conversation_history(self, client):
        """Test getting conversation history."""
        # Mock conversation history
        from datetime import datetime
        mock_messages = [
            Mock(role="user", content="Hello", timestamp=datetime.fromisoformat("2025-01-01T10:00:00"), metadata={}),
            Mock(role="assistant", content="Hi there!", timestamp=datetime.fromisoformat("2025-01-01T10:00:01"), metadata={})
        ]
        
        with patch('app.main.conversation_manager') as mock_manager:
            mock_manager.get_recent_history.return_value = mock_messages
            
            response = client.get("/conversation/test_user")
            assert response.status_code == 200
            
            data = response.json()
            assert data["user_id"] == "test_user"
            assert data["total_messages"] == 2
    
    def test_clear_conversation_history(self, client):
        """Test clearing conversation history."""
        with patch('app.main.conversation_manager') as mock_manager:
            response = client.delete("/conversation/test_user")
            assert response.status_code == 200
            
            data = response.json()
            assert data["user_id"] == "test_user"
            assert data["status"] == "success"
            
            mock_manager.clear_conversation.assert_called_once_with("test_user")