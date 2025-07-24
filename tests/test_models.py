"""
Tests for Pydantic models used in the API.
"""

import pytest
from pydantic import ValidationError
from app.models import (
    ChatQueryRequest,
    ChatQueryResponse,
    ConversationHistoryResponse,
    HealthResponse
)


class TestChatQueryRequest:
    """Test cases for ChatQueryRequest model."""
    
    def test_valid_chat_query_request(self):
        """Test creating a valid chat query request."""
        data = {
            "user_id": "alice123",
            "query": "What smartphones do you have?"
        }
        
        request = ChatQueryRequest(**data)
        
        assert request.user_id == "alice123"
        assert request.query == "What smartphones do you have?"
    
    def test_chat_query_request_with_minimal_valid_data(self):
        """Test chat query request with minimal valid data."""
        data = {
            "user_id": "a",  # Single character, should be valid
            "query": "x"     # Single character, should be valid
        }
        
        request = ChatQueryRequest(**data)
        
        assert request.user_id == "a"
        assert request.query == "x"
    
    def test_chat_query_request_with_max_query_length(self):
        """Test chat query request with maximum allowed query length."""
        data = {
            "user_id": "user123",
            "query": "x" * 1000  # Maximum allowed length
        }
        
        request = ChatQueryRequest(**data)
        
        assert request.user_id == "user123"
        assert len(request.query) == 1000
        assert request.query == "x" * 1000
    
    def test_chat_query_request_empty_user_id(self):
        """Test that empty user_id raises validation error."""
        data = {
            "user_id": "",
            "query": "Valid query"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ChatQueryRequest(**data)
        
        error = exc_info.value.errors()[0]
        assert error["type"] == "string_too_short"
        assert "user_id" in str(error["loc"])
    
    def test_chat_query_request_empty_query(self):
        """Test that empty query raises validation error."""
        data = {
            "user_id": "valid_user",
            "query": ""
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ChatQueryRequest(**data)
        
        error = exc_info.value.errors()[0]
        assert error["type"] == "string_too_short"
        assert "query" in str(error["loc"])
    
    def test_chat_query_request_query_too_long(self):
        """Test that query exceeding max length raises validation error."""
        data = {
            "user_id": "user123",
            "query": "x" * 1001  # Exceeds maximum length
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ChatQueryRequest(**data)
        
        error = exc_info.value.errors()[0]
        assert error["type"] == "string_too_long"
        assert "query" in str(error["loc"])
    
    def test_chat_query_request_missing_fields(self):
        """Test that missing required fields raise validation errors."""
        # Missing user_id
        with pytest.raises(ValidationError) as exc_info:
            ChatQueryRequest(query="Valid query")
        
        assert any("user_id" in str(error["loc"]) for error in exc_info.value.errors())
        
        # Missing query
        with pytest.raises(ValidationError) as exc_info:
            ChatQueryRequest(user_id="valid_user")
        
        assert any("query" in str(error["loc"]) for error in exc_info.value.errors())
        
        # Missing both fields
        with pytest.raises(ValidationError) as exc_info:
            ChatQueryRequest()
        
        errors = exc_info.value.errors()
        assert len(errors) == 2
        assert any("user_id" in str(error["loc"]) for error in errors)
        assert any("query" in str(error["loc"]) for error in errors)
    
    def test_chat_query_request_invalid_types(self):
        """Test that invalid field types raise validation errors."""
        # user_id as integer
        with pytest.raises(ValidationError) as exc_info:
            ChatQueryRequest(user_id=123, query="Valid query")
        
        error = exc_info.value.errors()[0]
        assert error["type"] == "string_type"
        assert "user_id" in str(error["loc"])
        
        # query as integer
        with pytest.raises(ValidationError) as exc_info:
            ChatQueryRequest(user_id="valid_user", query=123)
        
        error = exc_info.value.errors()[0]
        assert error["type"] == "string_type"
        assert "query" in str(error["loc"])
    
    def test_chat_query_request_special_characters(self):
        """Test chat query request with special characters."""
        data = {
            "user_id": "user@domain.com",
            "query": "What's the price of iPhone 15 Pro Max? 💰📱"
        }
        
        request = ChatQueryRequest(**data)
        
        assert request.user_id == "user@domain.com"
        assert request.query == "What's the price of iPhone 15 Pro Max? 💰📱"
    
    def test_chat_query_request_whitespace_handling(self):
        """Test how whitespace is handled in fields."""
        data = {
            "user_id": "  user123  ",  # Leading/trailing whitespace
            "query": "  What products do you have?  "
        }
        
        request = ChatQueryRequest(**data)
        
        # Pydantic doesn't strip whitespace by default
        assert request.user_id == "  user123  "
        assert request.query == "  What products do you have?  "


class TestChatQueryResponse:
    """Test cases for ChatQueryResponse model."""
    
    def test_valid_chat_query_response(self):
        """Test creating a valid chat query response."""
        data = {
            "user_id": "alice123",
            "query": "What smartphones do you have?",
            "response": "We have iPhone 15 Pro Max and Samsung Galaxy S24 Ultra.",
            "conversation_context": {
                "is_followup_question": False,
                "conversation_length": 2,
                "context_used": True
            },
            "search_enhancement": {
                "original_query": "What smartphones do you have?",
                "optimized_queries": ["smartphone options", "phone models"],
                "reasoning": "Generated multiple queries for better search",
                "context_usage": "Used conversation context"
            },
            "metadata": {
                "retrieval": {"top_k": 3, "total_results": 2},
                "generation": {"model": "claude-3-haiku-20240307", "tokens": 150},
                "documents_used": 2,
                "has_error": False,
                "conversation_aware": True
            }
        }
        
        response = ChatQueryResponse(**data)
        
        assert response.user_id == "alice123"
        assert response.query == "What smartphones do you have?"
        assert "iPhone 15 Pro Max" in response.response
        assert response.conversation_context["is_followup_question"] is False
        assert response.search_enhancement["optimized_queries"] == ["smartphone options", "phone models"]
        assert response.metadata["documents_used"] == 2
        assert response.debug_info is None
    
    def test_chat_query_response_with_debug_info(self):
        """Test chat query response with debug information."""
        data = {
            "user_id": "user123",
            "query": "test query",
            "response": "test response",
            "conversation_context": {},
            "search_enhancement": {},
            "metadata": {},
            "debug_info": {
                "error": "Something went wrong",
                "retrieved_documents": [],
                "conversation_history": "Previous messages"
            }
        }
        
        response = ChatQueryResponse(**data)
        
        assert response.debug_info is not None
        assert response.debug_info["error"] == "Something went wrong"
        assert response.debug_info["retrieved_documents"] == []
    
    def test_chat_query_response_missing_required_fields(self):
        """Test that missing required fields raise validation errors."""
        incomplete_data = {
            "user_id": "user123",
            # Missing other required fields
        }
        
        with pytest.raises(ValidationError) as exc_info:
            ChatQueryResponse(**incomplete_data)
        
        errors = exc_info.value.errors()
        required_fields = {"query", "response", "conversation_context", "search_enhancement", "metadata"}
        error_fields = {str(error["loc"][0]) for error in errors}
        
        assert required_fields.issubset(error_fields)
    
    def test_chat_query_response_invalid_types(self):
        """Test chat query response with invalid field types."""
        # conversation_context should be dict, not string
        with pytest.raises(ValidationError) as exc_info:
            ChatQueryResponse(
                user_id="user123",
                query="test",
                response="test",
                conversation_context="invalid",  # Should be dict
                search_enhancement={},
                metadata={}
            )
        
        error = exc_info.value.errors()[0]
        assert error["type"] == "dict_type"
        assert "conversation_context" in str(error["loc"])
    
    def test_chat_query_response_empty_dicts(self):
        """Test chat query response with empty dictionaries."""
        data = {
            "user_id": "user123",
            "query": "test query",
            "response": "test response",
            "conversation_context": {},
            "search_enhancement": {},
            "metadata": {}
        }
        
        response = ChatQueryResponse(**data)
        
        assert response.conversation_context == {}
        assert response.search_enhancement == {}
        assert response.metadata == {}


class TestConversationHistoryResponse:
    """Test cases for ConversationHistoryResponse model."""
    
    def test_valid_conversation_history_response(self):
        """Test creating a valid conversation history response."""
        data = {
            "user_id": "alice123",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello",
                    "timestamp": "2025-01-01T10:00:00",
                    "metadata": {}
                },
                {
                    "role": "assistant",
                    "content": "Hi there!",
                    "timestamp": "2025-01-01T10:00:01",
                    "metadata": {"documents_used": 0}
                }
            ],
            "total_messages": 2
        }
        
        response = ConversationHistoryResponse(**data)
        
        assert response.user_id == "alice123"
        assert len(response.messages) == 2
        assert response.total_messages == 2
        assert response.messages[0]["role"] == "user"
        assert response.messages[1]["role"] == "assistant"
    
    def test_conversation_history_response_empty_messages(self):
        """Test conversation history response with empty messages list."""
        data = {
            "user_id": "user123",
            "messages": [],
            "total_messages": 0
        }
        
        response = ConversationHistoryResponse(**data)
        
        assert response.user_id == "user123"
        assert response.messages == []
        assert response.total_messages == 0
    
    def test_conversation_history_response_mismatched_count(self):
        """Test conversation history with mismatched message count."""
        # This should still be valid - the model doesn't enforce consistency
        data = {
            "user_id": "user123",
            "messages": [{"role": "user", "content": "test"}],
            "total_messages": 5  # Doesn't match actual count
        }
        
        response = ConversationHistoryResponse(**data)
        
        assert len(response.messages) == 1
        assert response.total_messages == 5  # Model accepts the provided value
    
    def test_conversation_history_response_missing_fields(self):
        """Test that missing required fields raise validation errors."""
        with pytest.raises(ValidationError) as exc_info:
            ConversationHistoryResponse(user_id="user123")
        
        errors = exc_info.value.errors()
        error_fields = {str(error["loc"][0]) for error in errors}
        
        assert "messages" in error_fields
        assert "total_messages" in error_fields
    
    def test_conversation_history_response_invalid_types(self):
        """Test conversation history response with invalid field types."""
        # messages should be list, not string
        with pytest.raises(ValidationError) as exc_info:
            ConversationHistoryResponse(
                user_id="user123",
                messages="invalid",  # Should be list
                total_messages=0
            )
        
        error = exc_info.value.errors()[0]
        assert error["type"] == "list_type"
        assert "messages" in str(error["loc"])
        
        # total_messages should be int, not string
        with pytest.raises(ValidationError) as exc_info:
            ConversationHistoryResponse(
                user_id="user123",
                messages=[],
                total_messages="invalid"  # Should be int
            )
        
        error = exc_info.value.errors()[0]
        assert error["type"] in ["int_type", "int_parsing"]
        assert "total_messages" in str(error["loc"])


class TestHealthResponse:
    """Test cases for HealthResponse model."""
    
    def test_valid_health_response(self):
        """Test creating a valid health response."""
        data = {
            "status": "healthy",
            "version": "2.0.0",
            "vector_store_status": "healthy",
            "indexed_documents": 8
        }
        
        response = HealthResponse(**data)
        
        assert response.status == "healthy"
        assert response.version == "2.0.0"
        assert response.vector_store_status == "healthy"
        assert response.indexed_documents == 8
    
    def test_health_response_different_statuses(self):
        """Test health response with different status values."""
        status_combinations = [
            ("healthy", "healthy"),
            ("unhealthy", "error"),
            ("degraded", "empty"),
            ("maintenance", "initializing")
        ]
        
        for status, vector_status in status_combinations:
            data = {
                "status": status,
                "version": "1.0.0",
                "vector_store_status": vector_status,
                "indexed_documents": 5
            }
            
            response = HealthResponse(**data)
            
            assert response.status == status
            assert response.vector_store_status == vector_status
    
    def test_health_response_zero_documents(self):
        """Test health response with zero indexed documents."""
        data = {
            "status": "healthy",
            "version": "2.0.0",
            "vector_store_status": "empty",
            "indexed_documents": 0
        }
        
        response = HealthResponse(**data)
        
        assert response.indexed_documents == 0
        assert response.vector_store_status == "empty"
    
    def test_health_response_large_document_count(self):
        """Test health response with large document count."""
        data = {
            "status": "healthy",
            "version": "2.0.0",
            "vector_store_status": "healthy",
            "indexed_documents": 1000000
        }
        
        response = HealthResponse(**data)
        
        assert response.indexed_documents == 1000000
    
    def test_health_response_missing_fields(self):
        """Test that missing required fields raise validation errors."""
        with pytest.raises(ValidationError) as exc_info:
            HealthResponse()
        
        errors = exc_info.value.errors()
        required_fields = {"status", "version", "vector_store_status", "indexed_documents"}
        error_fields = {str(error["loc"][0]) for error in errors}
        
        assert required_fields == error_fields
    
    def test_health_response_invalid_types(self):
        """Test health response with invalid field types."""
        # indexed_documents should be int, not string
        with pytest.raises(ValidationError) as exc_info:
            HealthResponse(
                status="healthy",
                version="2.0.0",
                vector_store_status="healthy",
                indexed_documents="invalid"  # Should be int
            )
        
        error = exc_info.value.errors()[0]
        assert error["type"] in ["int_type", "int_parsing"]
        assert "indexed_documents" in str(error["loc"])
    
    def test_health_response_negative_document_count(self):
        """Test health response with negative document count."""
        # This should be valid - model doesn't restrict to positive integers
        data = {
            "status": "error",
            "version": "2.0.0",
            "vector_store_status": "error",
            "indexed_documents": -1
        }
        
        response = HealthResponse(**data)
        
        assert response.indexed_documents == -1


class TestModelSerialization:
    """Test serialization and deserialization of models."""
    
    def test_chat_query_request_serialization(self):
        """Test ChatQueryRequest serialization to dict and JSON."""
        request = ChatQueryRequest(
            user_id="test_user",
            query="What laptops do you have?"
        )
        
        # Test dict conversion
        data = request.model_dump()
        assert data["user_id"] == "test_user"
        assert data["query"] == "What laptops do you have?"
        
        # Test JSON conversion
        json_str = request.model_dump_json()
        assert "test_user" in json_str
        assert "What laptops do you have?" in json_str
    
    def test_chat_query_response_serialization(self):
        """Test ChatQueryResponse serialization with complex nested data."""
        response = ChatQueryResponse(
            user_id="test_user",
            query="test query",
            response="test response",
            conversation_context={"is_followup": False, "length": 1},
            search_enhancement={"queries": ["q1", "q2"]},
            metadata={"docs": 2, "tokens": 100}
        )
        
        data = response.model_dump()
        assert data["conversation_context"]["is_followup"] is False
        assert data["search_enhancement"]["queries"] == ["q1", "q2"]
        assert data["metadata"]["docs"] == 2
    
    def test_model_round_trip_serialization(self):
        """Test that models can be serialized and deserialized correctly."""
        original_request = ChatQueryRequest(
            user_id="roundtrip_user",
            query="Test round trip serialization"
        )
        
        # Serialize to dict
        data = original_request.model_dump()
        
        # Deserialize back to model
        recreated_request = ChatQueryRequest(**data)
        
        # Verify they're identical
        assert original_request.user_id == recreated_request.user_id
        assert original_request.query == recreated_request.query
        assert original_request == recreated_request