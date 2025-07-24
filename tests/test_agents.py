import pytest
from unittest.mock import Mock, patch
from app.agents.retriever_agent import RetrieverAgent
from app.agents.responder_agent import ResponderAgent
from app.vector_store import VectorStore


class TestRetrieverAgent:
    
    def test_retriever_agent_initialization(self):
        mock_vector_store = Mock(spec=VectorStore)
        agent = RetrieverAgent(mock_vector_store)
        
        assert agent.vector_store == mock_vector_store
        assert agent.name == "retriever_agent"
    
    @patch('app.agents.retriever_agent.anthropic.Anthropic')
    def test_retrieve_with_default_top_k(self, mock_anthropic):
        # Mock the LLM response for query optimization
        mock_response = Mock()
        mock_response.content = [Mock(text='{"search_queries": ["iPhone camera"], "reasoning": "test", "context_used": "none"}')]
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        mock_vector_store = Mock(spec=VectorStore)
        mock_results = [
            {"content": "iPhone 15 Pro", "metadata": {"source": "iphone.txt"}, "score": 0.95, "rank": 1}
        ]
        mock_vector_store.search.return_value = mock_results
        
        agent = RetrieverAgent(mock_vector_store)
        result = agent.retrieve("iPhone camera")
        
        assert result["original_query"] == "iPhone camera"
        assert result["retrieved_documents"] == mock_results
        assert "retrieval_metadata" in result
        mock_vector_store.search.assert_called()
    
    @patch('app.agents.retriever_agent.anthropic.Anthropic')
    def test_retrieve_with_custom_top_k(self, mock_anthropic):
        # Mock the LLM response
        mock_response = Mock()
        mock_response.content = [Mock(text='{"search_queries": ["test query"], "reasoning": "test", "context_used": "none"}')]
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        mock_vector_store = Mock(spec=VectorStore)
        mock_vector_store.search.return_value = []
        
        agent = RetrieverAgent(mock_vector_store)
        agent.retrieve("test query", conversation_history="", top_k=5)
        
        mock_vector_store.search.assert_called_with("test query", top_k=5)


class TestResponderAgent:
    
    @patch('app.agents.responder_agent.anthropic.Anthropic')
    def test_responder_agent_initialization(self, mock_anthropic):
        agent = ResponderAgent()
        assert agent.name == "responder_agent"
        mock_anthropic.assert_called_once()
    
    @patch('app.agents.responder_agent.anthropic.Anthropic')
    def test_generate_response_success(self, mock_anthropic):
        # Mock the Anthropic response
        mock_response = Mock()
        mock_response.content = [Mock(text="The iPhone 15 Pro Max features advanced camera technology.")]
        mock_response.usage = Mock(input_tokens=100, output_tokens=50)
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        agent = ResponderAgent()
        
        retrieved_docs = [
            {
                "content": "iPhone 15 Pro Max with 48MP camera",
                "metadata": {"source": "iphone.txt"},
                "score": 0.95,
                "rank": 1
            }
        ]
        
        result = agent.generate_response("What camera does iPhone 15 Pro Max have?", retrieved_docs)
        
        assert result["query"] == "What camera does iPhone 15 Pro Max have?"
        assert result["response"] == "The iPhone 15 Pro Max features advanced camera technology."
        assert "retrieved_context" in result
        assert result["generation_metadata"]["documents_used"] == 1
        assert result["generation_metadata"]["input_tokens"] == 100
        assert result["generation_metadata"]["output_tokens"] == 50
        mock_client.messages.create.assert_called_once()
    
    @patch('app.agents.responder_agent.anthropic.Anthropic')
    def test_generate_response_with_llm_error(self, mock_anthropic):
        # Mock Anthropic client to raise an exception
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.return_value = mock_client
        
        agent = ResponderAgent()
        
        retrieved_docs = [
            {"content": "Test content", "metadata": {}, "score": 0.8, "rank": 1}
        ]
        
        result = agent.generate_response("test query", retrieved_docs)
        
        assert "error" in result
        assert result["generation_metadata"]["error"] is True
        assert "I apologize, but I encountered an error" in result["response"]
    
    @patch('app.agents.responder_agent.anthropic.Anthropic')
    def test_generate_response_empty_retrieved_docs(self, mock_anthropic):
        mock_response = Mock()
        mock_response.content = [Mock(text="I don't have specific information about that.")]
        mock_response.usage = Mock(input_tokens=50, output_tokens=20)
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        agent = ResponderAgent()
        
        result = agent.generate_response("test query", [])
        
        assert result["generation_metadata"]["documents_used"] == 0
        assert result["retrieved_context"] == "No relevant documents found."
    
    @patch('app.agents.responder_agent.anthropic.Anthropic')
    def test_generate_response_multiple_documents(self, mock_anthropic):
        mock_response = Mock()
        mock_response.content = [Mock(text="Based on the products available...")]
        mock_response.usage = Mock(input_tokens=200, output_tokens=100)
        
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        agent = ResponderAgent()
        
        retrieved_docs = [
            {"content": "iPhone specs", "metadata": {}, "score": 0.95, "rank": 1},
            {"content": "Samsung specs", "metadata": {}, "score": 0.85, "rank": 2},
            {"content": "Google Pixel specs", "metadata": {}, "score": 0.75, "rank": 3}
        ]
        
        result = agent.generate_response("Compare smartphones", retrieved_docs)
        
        assert result["generation_metadata"]["documents_used"] == 3
        assert "Document 1" in result["retrieved_context"]
        assert "Document 2" in result["retrieved_context"]
        assert "Document 3" in result["retrieved_context"]