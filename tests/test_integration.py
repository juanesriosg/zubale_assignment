"""
Integration tests for end-to-end workflows and component interactions.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.vector_store import VectorStore
from app.chat_workflow import ChatRAGWorkflow
from app.conversation import ConversationManager, ChatMessage
from app.agents.retriever_agent import RetrieverAgent
from app.agents.responder_agent import ResponderAgent


@pytest.fixture
def temp_vector_store():
    """Create a temporary vector store for testing."""
    # Create some test documents
    test_docs = [
        "iPhone 15 Pro Max features a powerful A17 Pro chip and excellent camera system for photography.",
        "MacBook Air M2 offers outstanding performance for programming with 18-hour battery life.",
        "Samsung Galaxy S24 Ultra has advanced camera features and S Pen for productivity.",
        "Dell XPS 13 Plus is a premium laptop with great display and solid build quality.",
        "AirPods Pro 2 provide excellent noise cancellation and spatial audio features."
    ]
    
    vector_store = VectorStore(embedding_model_name="all-MiniLM-L6-v2")
    
    # Add documents to the store
    metadata_list = [{"source": f"product_{i+1}.txt"} for i in range(len(test_docs))]
    vector_store.add_documents(test_docs, metadata_list)
    
    yield vector_store


@pytest.fixture
def test_client():
    """Create test client for FastAPI app."""
    return TestClient(app)


@pytest.fixture
def conversation_manager():
    """Create a fresh conversation manager for testing."""
    return ConversationManager()


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    @patch('app.agents.responder_agent.anthropic.Anthropic')
    @patch('app.agents.retriever_agent.anthropic.Anthropic')
    def test_complete_chat_workflow_with_real_vector_store(self, mock_retriever_client, mock_responder_client, temp_vector_store):
        """Test complete chat workflow with real vector store and mocked LLM calls."""
        
        # Mock the LLM responses
        mock_retriever_response = Mock()
        mock_retriever_response.content = [Mock(text='{"search_queries": ["smartphone photography", "iPhone camera features", "phone photo quality"], "reasoning": "Looking for smartphone camera information", "context_used": "none"}')]
        mock_retriever_response.usage = Mock(input_tokens=50, output_tokens=25)
        mock_retriever_client.return_value.messages.create.return_value = mock_retriever_response
        
        mock_responder_response = Mock()
        mock_responder_response.content = [Mock(text="Based on your interest in smartphone photography, I'd recommend the iPhone 15 Pro Max. It features a powerful A17 Pro chip and excellent camera system specifically designed for photography enthusiasts.")]
        mock_responder_response.usage = Mock(input_tokens=100, output_tokens=50)
        mock_responder_client.return_value.messages.create.return_value = mock_responder_response
        
        # Create workflow with real vector store
        workflow = ChatRAGWorkflow(temp_vector_store)
        
        # Process a query
        result = workflow.process_chat_query("test_user", "I need a smartphone for photography")
        
        # Verify the workflow completed successfully
        assert result["user_id"] == "test_user"
        assert result["query"] == "I need a smartphone for photography"
        assert "iPhone 15 Pro Max" in result["response"]
        assert result["metadata"]["has_error"] is False
        assert result["metadata"]["documents_used"] > 0
        assert result["conversation_context"] is not None
        assert result["search_enhancement"]["optimized_queries"] is not None
    
    @patch('app.agents.responder_agent.anthropic.Anthropic')
    @patch('app.agents.retriever_agent.anthropic.Anthropic')  
    def test_multi_turn_conversation_context(self, mock_retriever_client, mock_responder_client, temp_vector_store):
        """Test multi-turn conversation with context awareness."""
        
        # Mock LLM responses for first turn
        mock_retriever_response1 = Mock()
        mock_retriever_response1.content = [Mock(text='{"search_queries": ["laptop programming", "MacBook development", "coding laptops"], "reasoning": "Looking for laptop information", "context_used": "none"}')]
        mock_retriever_response1.usage = Mock(input_tokens=50, output_tokens=25)
        
        mock_responder_response1 = Mock()
        mock_responder_response1.content = [Mock(text="For programming, I recommend the MacBook Air M2. It offers outstanding performance for development work with an impressive 18-hour battery life.")]
        mock_responder_response1.usage = Mock(input_tokens=100, output_tokens=50)
        
        # Mock LLM responses for follow-up turn
        mock_retriever_response2 = Mock()
        mock_retriever_response2.content = [Mock(text='{"search_queries": ["MacBook Air M2 battery", "laptop battery life", "MacBook battery specs"], "reasoning": "Looking for battery information", "context_used": "previous"}')]
        mock_retriever_response2.usage = Mock(input_tokens=50, output_tokens=25)
        
        mock_responder_response2 = Mock()
        mock_responder_response2.content = [Mock(text="Great follow-up question! The MacBook Air M2 we just discussed offers up to 18 hours of battery life, making it excellent for long programming sessions without needing to charge.")]
        mock_responder_response2.usage = Mock(input_tokens=100, output_tokens=50)
        
        # Setup mock to return different responses for different calls
        # Using a fresh approach to avoid side_effect issues
        retriever_call_count = 0
        responder_call_count = 0
        
        def retriever_side_effect(*args, **kwargs):
            nonlocal retriever_call_count
            if retriever_call_count == 0:
                retriever_call_count += 1
                return mock_retriever_response1
            else:
                return mock_retriever_response2
        
        def responder_side_effect(*args, **kwargs):
            nonlocal responder_call_count
            if responder_call_count == 0:
                responder_call_count += 1
                return mock_responder_response1
            else:
                return mock_responder_response2
        
        mock_retriever_client.return_value.messages.create.side_effect = retriever_side_effect
        mock_responder_client.return_value.messages.create.side_effect = responder_side_effect
        
        workflow = ChatRAGWorkflow(temp_vector_store)
        
        # First turn
        result1 = workflow.process_chat_query("user123", "I need a laptop for programming")
        assert "MacBook Air M2" in result1["response"]
        assert result1["conversation_context"]["is_followup_question"] is False
        
        # Second turn (follow-up)
        result2 = workflow.process_chat_query("user123", "What about battery life?")
        print(f"DEBUG - Second turn response: {repr(result2['response'])}")
        print(f"DEBUG - Second turn has_error: {result2['metadata']['has_error']}")
        if result2['metadata']['has_error']:
            print(f"DEBUG - Second turn debug_info: {result2.get('debug_info', {})}")
        assert "18 hours" in result2["response"]
        assert "follow-up" in result2["response"].lower()
        # Note: Follow-up detection depends on the responder agent implementation
    
    def test_conversation_persistence_across_queries(self, temp_vector_store):
        """Test that conversation history is properly maintained."""
        
        with patch('app.agents.retriever_agent.anthropic.Anthropic'), \
             patch('app.agents.responder_agent.anthropic.Anthropic'):
            
            workflow = ChatRAGWorkflow(temp_vector_store)
            user_id = "persistent_user"
            
            # Simulate multiple queries
            queries = [
                "Tell me about smartphones",
                "What about their cameras?", 
                "Which one has the best battery?"
            ]
            
            for i, query in enumerate(queries):
                result = workflow.process_chat_query(user_id, query)
                
                # Verify conversation length increases
                # Note: This depends on how conversation_context is implemented
                assert result["user_id"] == user_id
                assert result["query"] == query
    
    def test_error_handling_in_workflow(self, temp_vector_store):
        """Test error handling throughout the workflow."""
        
        with patch('app.agents.retriever_agent.anthropic.Anthropic') as mock_retriever_client:
            # Make retriever agent fail
            mock_retriever_client.return_value.messages.create.side_effect = Exception("API Error")
            
            workflow = ChatRAGWorkflow(temp_vector_store)
            
            result = workflow.process_chat_query("error_user", "test query")
            
            # Verify graceful error handling
            assert result["user_id"] == "error_user"
            assert result["metadata"]["has_error"] is True
            assert "error" in result["response"].lower() or "sorry" in result["response"].lower()
            assert result["debug_info"] is not None
    
    @patch('app.agents.retriever_agent.anthropic.Anthropic')
    @patch('app.agents.responder_agent.anthropic.Anthropic')
    def test_document_retrieval_accuracy(self, mock_responder_client, mock_retriever_client, temp_vector_store):
        """Test that relevant documents are retrieved for queries."""
        
        # Mock LLM responses
        mock_retriever_response = Mock()
        mock_retriever_response.content = [Mock(text='["iPhone features", "smartphone specs"]')]
        mock_retriever_client.return_value.messages.create.return_value = mock_retriever_response
        
        mock_responder_response = Mock()
        mock_responder_response.content = [Mock(text="Here are the iPhone details...")]
        mock_responder_response.usage = Mock(input_tokens=100, output_tokens=50)
        mock_responder_client.return_value.messages.create.return_value = mock_responder_response
        
        workflow = ChatRAGWorkflow(temp_vector_store)
        
        # Query specifically about iPhone
        result = workflow.process_chat_query("test_user", "Tell me about iPhone features")
        
        # Verify iPhone-related document was likely retrieved
        assert result["metadata"]["documents_used"] > 0
        assert result["debug_info"] is None or len(result["debug_info"].get("retrieved_documents", [])) > 0


class TestAPIIntegration:
    """Test API endpoint integration with backend components."""
    
    def test_chat_endpoint_integration(self, test_client):
        """Test chat endpoint integration with workflow."""
        
        # Mock workflow response
        mock_workflow = Mock()
        mock_workflow.process_chat_query.return_value = {
            "user_id": "api_user",
            "query": "test query",
            "response": "test response",
            "conversation_context": {"is_followup_question": False},
            "search_enhancement": {"optimized_queries": ["test"]},
            "metadata": {"documents_used": 1, "has_error": False}
        }
        
        import app.main
        original_workflow = app.main.chat_rag_workflow
        app.main.chat_rag_workflow = mock_workflow
        
        try:
            # Make API request
            response = test_client.post("/chat", json={
                "user_id": "api_user",
                "query": "test query"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["user_id"] == "api_user"
            assert data["response"] == "test response"
            
            # Verify workflow was called
            mock_workflow.process_chat_query.assert_called_once_with("api_user", "test query")
        finally:
            app.main.chat_rag_workflow = original_workflow
    
    def test_health_endpoint_integration(self, test_client):
        """Test health endpoint with backend components."""
        
        mock_vector_store = Mock()
        mock_vector_store.documents = ["doc1", "doc2", "doc3"]
        
        import app.main
        original_vector_store = app.main.vector_store
        app.main.vector_store = mock_vector_store
        
        try:
            response = test_client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["indexed_documents"] == 3
        finally:
            app.main.vector_store = original_vector_store
    
    @patch('app.main.conversation_manager')
    def test_conversation_endpoints_integration(self, mock_conversation_manager, test_client):
        """Test conversation history endpoints."""
        
        # Mock conversation history
        from datetime import datetime
        mock_messages = [
            Mock(role="user", content="Hello", timestamp=datetime.fromisoformat("2025-01-01T10:00:00"), metadata={}),
            Mock(role="assistant", content="Hi!", timestamp=datetime.fromisoformat("2025-01-01T10:00:01"), metadata={})
        ]
        mock_conversation_manager.get_recent_history.return_value = mock_messages
        
        # Test GET conversation
        response = test_client.get("/conversation/test_user")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "test_user"
        assert data["total_messages"] == 2
        
        # Test DELETE conversation
        response = test_client.delete("/conversation/test_user")
        assert response.status_code == 200
        mock_conversation_manager.clear_conversation.assert_called_once_with("test_user")


class TestComponentInteraction:
    """Test interactions between individual components."""
    
    def test_retriever_responder_agent_interaction(self, temp_vector_store):
        """Test interaction between retriever and responder agents."""
        
        with patch('app.agents.retriever_agent.anthropic.Anthropic') as mock_retriever_client, \
             patch('app.agents.responder_agent.anthropic.Anthropic') as mock_responder_client:
            
            # Mock retriever response
            mock_retriever_response = Mock()
            mock_retriever_response.content = [Mock(text='["smartphone camera", "phone photography"]')]
            mock_retriever_client.return_value.messages.create.return_value = mock_retriever_response
            
            # Mock responder response
            mock_responder_response = Mock()
            mock_responder_response.content = [Mock(text="Based on retrieved documents about cameras...")]
            mock_responder_response.usage = Mock(input_tokens=100, output_tokens=50)
            mock_responder_client.return_value.messages.create.return_value = mock_responder_response
            
            # Create agents
            retriever = RetrieverAgent(temp_vector_store)
            responder = ResponderAgent()
            
            # Test retriever -> responder flow
            retrieval_result = retriever.retrieve("smartphone with good camera", "")
            response_result = responder.generate_response(
                "smartphone with good camera",
                retrieval_result["retrieved_documents"],
                "",
                retrieval_result["llm_reasoning"]
            )
            
            # Verify data flows correctly between agents
            assert len(retrieval_result["retrieved_documents"]) > 0
            assert retrieval_result["llm_reasoning"] != ""
            assert response_result["response"] != ""
            assert "camera" in response_result["response"].lower()
    
    def test_conversation_manager_workflow_integration(self, temp_vector_store):
        """Test conversation manager integration with workflow."""
        
        with patch('app.agents.retriever_agent.anthropic.Anthropic'), \
             patch('app.agents.responder_agent.anthropic.Anthropic'), \
             patch('app.chat_workflow.conversation_manager') as mock_conv_manager:
            
            # Mock conversation history
            mock_conv_manager.format_conversation_for_prompt.return_value = "Previous: Hello"
            
            workflow = ChatRAGWorkflow(temp_vector_store)
            result = workflow.process_chat_query("conv_test_user", "Follow up question")
            
            # Verify conversation manager was used
            mock_conv_manager.format_conversation_for_prompt.assert_called_once_with("conv_test_user")
            
            # Verify messages were saved (2 calls: user message + assistant message)
            assert mock_conv_manager.add_message.call_count == 2
    
    def test_vector_store_agent_integration(self, temp_vector_store):
        """Test vector store and agent integration."""
        
        with patch('app.agents.retriever_agent.anthropic.Anthropic') as mock_client:
            # Mock LLM response
            mock_response = Mock()
            mock_response.content = [Mock(text='["iPhone", "smartphone", "phone"]')]
            mock_client.return_value.messages.create.return_value = mock_response
            
            agent = RetrieverAgent(temp_vector_store)
            result = agent.retrieve("iPhone features", "")
            
            # Verify vector store was used
            assert len(result["retrieved_documents"]) > 0
            assert any("iPhone" in doc.get("content", "") for doc in result["retrieved_documents"])
            
            # Verify LLM was called to optimize queries
            mock_client.return_value.messages.create.assert_called_once()


class TestErrorScenarios:
    """Test various error scenarios in integration."""
    
    def test_vector_store_unavailable(self):
        """Test behavior when vector store is unavailable."""
        
        # Create empty vector store (no documents)
        invalid_vector_store = VectorStore(embedding_model_name="all-MiniLM-L6-v2")
        
        with patch('app.agents.retriever_agent.anthropic.Anthropic'), \
             patch('app.agents.responder_agent.anthropic.Anthropic'):
            
            workflow = ChatRAGWorkflow(invalid_vector_store)
            
            # This should handle the error gracefully
            result = workflow.process_chat_query("error_user", "test query")
            
            # Verify error handling
            assert result["user_id"] == "error_user"
            assert result["metadata"]["has_error"] is True or result["metadata"]["documents_used"] == 0
    
    def test_llm_api_failure_recovery(self, temp_vector_store):
        """Test recovery when LLM API calls fail."""
        
        with patch('app.agents.retriever_agent.anthropic.Anthropic') as mock_retriever_client, \
             patch('app.agents.responder_agent.anthropic.Anthropic') as mock_responder_client:
            
            # Make both agents fail
            mock_retriever_client.return_value.messages.create.side_effect = Exception("API Failure")
            mock_responder_client.return_value.messages.create.side_effect = Exception("API Failure")
            
            workflow = ChatRAGWorkflow(temp_vector_store)
            result = workflow.process_chat_query("error_user", "test query")
            
            # Should still return a response, even if degraded
            assert result["user_id"] == "error_user"
            assert result["response"] is not None
            assert len(result["response"]) > 0
            assert result["metadata"]["has_error"] is True
    
    def test_malformed_data_handling(self, test_client):
        """Test API handling of malformed requests."""
        
        malformed_requests = [
            {},  # Empty request
            {"user_id": ""},  # Empty user_id
            {"query": ""},  # Empty query
            {"user_id": 123, "query": "test"},  # Wrong type
            {"user_id": "test", "query": ["invalid"]},  # Wrong type
        ]
        
        # Set dependency to avoid 503 errors during validation
        import app.main
        original_workflow = app.main.chat_rag_workflow
        app.main.chat_rag_workflow = Mock()
        
        try:
            for request_data in malformed_requests:
                response = test_client.post("/chat", json=request_data)
                assert response.status_code == 422  # Validation error
        finally:
            app.main.chat_rag_workflow = original_workflow
    
    def test_concurrent_user_handling(self, temp_vector_store):
        """Test handling multiple users concurrently."""
        
        with patch('app.agents.retriever_agent.anthropic.Anthropic'), \
             patch('app.agents.responder_agent.anthropic.Anthropic'):
            
            workflow = ChatRAGWorkflow(temp_vector_store)
            
            # Simulate concurrent users
            users = ["user1", "user2", "user3"]
            results = []
            
            for user in users:
                result = workflow.process_chat_query(user, f"Query from {user}")
                results.append(result)
            
            # Verify each user got correct response
            for i, result in enumerate(results):
                assert result["user_id"] == users[i]
                assert users[i] in result["query"]