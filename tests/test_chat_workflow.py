"""
Comprehensive tests for the ChatRAGWorkflow class.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from app.chat_workflow import ChatRAGWorkflow, ChatWorkflowState
from app.vector_store import VectorStore
from app.agents.retriever_agent import RetrieverAgent
from app.agents.responder_agent import ResponderAgent


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing."""
    mock_store = Mock(spec=VectorStore)
    mock_store.documents = ["doc1", "doc2", "doc3"]
    return mock_store


@pytest.fixture
def mock_retriever_agent():
    """Mock retriever agent with successful response."""
    mock_agent = Mock(spec=RetrieverAgent)
    mock_agent.retrieve.return_value = {
        "original_query": "test query",
        "optimized_queries": ["optimized query 1", "optimized query 2"],
        "llm_reasoning": "Generated multiple queries for better search",
        "context_usage": "Used conversation context to optimize queries",
        "retrieved_documents": [
            {"content": "Test document 1", "metadata": {"source": "doc1"}},
            {"content": "Test document 2", "metadata": {"source": "doc2"}}
        ],
        "retrieval_metadata": {"top_k": 3, "total_results": 2}
    }
    return mock_agent


@pytest.fixture
def mock_responder_agent():
    """Mock responder agent with successful response."""
    mock_agent = Mock(spec=ResponderAgent)
    mock_agent.generate_response.return_value = {
        "response": "This is a test response based on the retrieved documents.",
        "conversation_context": {
            "is_followup_question": False,
            "conversation_length": 2,
            "context_used": True
        },
        "retrieved_context": "Context from retrieved documents",
        "search_reasoning": "Search reasoning explanation",
        "generation_metadata": {
            "model": "claude-3-haiku-20240307",
            "tokens_used": 150,
            "documents_used": 2
        }
    }
    return mock_agent


@pytest.fixture
def workflow_with_mocks(mock_vector_store, mock_retriever_agent, mock_responder_agent):
    """Create workflow instance with mocked dependencies."""
    with patch('app.chat_workflow.RetrieverAgent', return_value=mock_retriever_agent), \
         patch('app.chat_workflow.ResponderAgent', return_value=mock_responder_agent):
        workflow = ChatRAGWorkflow(mock_vector_store)
        workflow.retriever_agent = mock_retriever_agent
        workflow.responder_agent = mock_responder_agent
        return workflow


@pytest.fixture
def sample_state():
    """Sample workflow state for testing."""
    return ChatWorkflowState(
        user_id="test_user",
        query="What smartphones do you have?",
        conversation_history="Previous conversation context",
        original_query="",
        optimized_queries=[],
        llm_reasoning="",
        context_usage="",
        retrieved_documents=[],
        retrieval_metadata={},
        response="",
        conversation_context={},
        retrieved_context="",
        search_reasoning="",
        generation_metadata={},
        error=""
    )


class TestChatRAGWorkflow:
    """Test cases for ChatRAGWorkflow class."""
    
    def test_workflow_initialization(self, mock_vector_store):
        """Test workflow initialization with dependencies."""
        with patch('app.chat_workflow.RetrieverAgent') as mock_retriever_cls, \
             patch('app.chat_workflow.ResponderAgent') as mock_responder_cls:
            
            workflow = ChatRAGWorkflow(mock_vector_store)
            
            # Verify agents are initialized
            mock_retriever_cls.assert_called_once_with(mock_vector_store)
            mock_responder_cls.assert_called_once()
            
            # Verify workflow is built
            assert workflow.workflow is not None
            assert hasattr(workflow, 'retriever_agent')
            assert hasattr(workflow, 'responder_agent')
    
    def test_build_workflow_structure(self, mock_vector_store):
        """Test that workflow graph is built correctly."""
        with patch('app.chat_workflow.RetrieverAgent'), \
             patch('app.chat_workflow.ResponderAgent'):
            
            workflow = ChatRAGWorkflow(mock_vector_store)
            
            # Verify workflow is compiled
            assert workflow.workflow is not None
            # Note: StateGraph internal structure testing is limited,
            # but we can verify it was created without errors
    
    def test_retrieve_node_success(self, workflow_with_mocks, sample_state, mock_retriever_agent):
        """Test successful retrieval node execution."""
        result_state = workflow_with_mocks._retrieve_node(sample_state)
        
        # Verify retriever agent was called correctly
        mock_retriever_agent.retrieve.assert_called_once_with(
            current_query="What smartphones do you have?",
            conversation_history="Previous conversation context"
        )
        
        # Verify state was updated correctly
        assert result_state["original_query"] == "test query"
        assert result_state["optimized_queries"] == ["optimized query 1", "optimized query 2"]
        assert result_state["llm_reasoning"] == "Generated multiple queries for better search"
        assert result_state["context_usage"] == "Used conversation context to optimize queries"
        assert len(result_state["retrieved_documents"]) == 2
        assert result_state["retrieval_metadata"]["total_results"] == 2
        assert result_state["error"] == ""
    
    def test_retrieve_node_error_handling(self, workflow_with_mocks, sample_state, mock_retriever_agent):
        """Test retrieval node error handling."""
        # Make retriever agent raise an exception
        mock_retriever_agent.retrieve.side_effect = Exception("Retrieval failed")
        
        result_state = workflow_with_mocks._retrieve_node(sample_state)
        
        # Verify error handling
        assert "Retrieval error: Retrieval failed" in result_state["error"]
        assert result_state["retrieved_documents"] == []
        assert result_state["retrieval_metadata"]["error"] is True
        assert result_state["llm_reasoning"] == "Error in retrieval processing"
        assert result_state["context_usage"] == "Error"
    
    def test_respond_node_success(self, workflow_with_mocks, mock_responder_agent):
        """Test successful response node execution."""
        # Setup state with retrieved documents
        state = ChatWorkflowState(
            user_id="test_user",
            query="What smartphones do you have?",
            conversation_history="Previous conversation",
            original_query="test query",
            optimized_queries=["query1"],
            llm_reasoning="Search reasoning",
            context_usage="Used context",
            retrieved_documents=[{"content": "doc1"}, {"content": "doc2"}],
            retrieval_metadata={"total_results": 2},
            response="",
            conversation_context={},
            retrieved_context="",
            search_reasoning="",
            generation_metadata={},
            error=""
        )
        
        result_state = workflow_with_mocks._respond_node(state)
        
        # Verify responder agent was called correctly
        mock_responder_agent.generate_response.assert_called_once_with(
            current_query="What smartphones do you have?",
            retrieved_docs=[{"content": "doc1"}, {"content": "doc2"}],
            conversation_history="Previous conversation",
            search_reasoning="Search reasoning"
        )
        
        # Verify state was updated correctly
        assert result_state["response"] == "This is a test response based on the retrieved documents."
        assert result_state["conversation_context"]["is_followup_question"] is False
        assert result_state["conversation_context"]["context_used"] is True
        assert result_state["generation_metadata"]["documents_used"] == 2
    
    def test_respond_node_with_retrieval_error(self, workflow_with_mocks, sample_state):
        """Test response node when retrieval failed."""
        # Setup state with retrieval error
        sample_state["error"] = "Retrieval error occurred"
        sample_state["retrieved_documents"] = []
        
        result_state = workflow_with_mocks._respond_node(sample_state)
        
        # Verify fallback response
        assert "I apologize, but I encountered an issue retrieving" in result_state["response"]
        assert result_state["generation_metadata"]["error"] is True
        assert result_state["conversation_context"]["is_followup_question"] is False
        assert result_state["conversation_context"]["context_used"] is False
    
    def test_respond_node_error_handling(self, workflow_with_mocks, mock_responder_agent):
        """Test response node error handling."""
        # Setup state with documents
        state = ChatWorkflowState(
            user_id="test_user",
            query="test query",
            conversation_history="history",
            original_query="",
            optimized_queries=[],
            llm_reasoning="reasoning",
            context_usage="",
            retrieved_documents=[{"content": "doc1"}],
            retrieval_metadata={},
            response="",
            conversation_context={},
            retrieved_context="",
            search_reasoning="",
            generation_metadata={},
            error=""
        )
        
        # Make responder agent raise an exception
        mock_responder_agent.generate_response.side_effect = Exception("Response generation failed")
        
        result_state = workflow_with_mocks._respond_node(state)
        
        # Verify error handling
        assert "Response generation error: Response generation failed" in result_state["error"]
        assert "I apologize, but I encountered an issue generating a response" in result_state["response"]
        assert result_state["generation_metadata"]["error"] is True
    
    @patch('app.chat_workflow.conversation_manager')
    def test_save_conversation_node_success(self, mock_conversation_manager, workflow_with_mocks):
        """Test successful conversation saving."""
        state = ChatWorkflowState(
            user_id="test_user",
            query="test query",
            conversation_history="",
            original_query="",
            optimized_queries=["opt1", "opt2"],
            llm_reasoning="LLM reasoning",
            context_usage="",
            retrieved_documents=[{"content": "doc1"}, {"content": "doc2"}],
            retrieval_metadata={},
            response="Test response",
            conversation_context={"is_followup_question": True},
            retrieved_context="",
            search_reasoning="",
            generation_metadata={"tokens": 100},
            error=""
        )
        
        result_state = workflow_with_mocks._save_conversation_node(state)
        
        # Verify conversation manager calls
        assert mock_conversation_manager.add_message.call_count == 2
        
        # Verify user message was saved
        user_call = mock_conversation_manager.add_message.call_args_list[0]
        assert user_call[1]["user_id"] == "test_user"
        assert user_call[1]["role"] == "user"
        assert user_call[1]["content"] == "test query"
        assert user_call[1]["metadata"]["is_followup"] is True
        assert user_call[1]["metadata"]["optimized_queries"] == ["opt1", "opt2"]
        
        # Verify assistant message was saved
        assistant_call = mock_conversation_manager.add_message.call_args_list[1]
        assert assistant_call[1]["user_id"] == "test_user"
        assert assistant_call[1]["role"] == "assistant"
        assert assistant_call[1]["content"] == "Test response"
        assert assistant_call[1]["metadata"]["documents_used"] == 2
        assert assistant_call[1]["metadata"]["search_reasoning"] == "LLM reasoning"
        
        # State should be unchanged on success
        assert result_state == state
    
    @patch('app.chat_workflow.conversation_manager')
    def test_save_conversation_node_error_handling(self, mock_conversation_manager, workflow_with_mocks, sample_state):
        """Test conversation saving error handling."""
        # Make conversation manager raise an exception
        mock_conversation_manager.add_message.side_effect = Exception("Save failed")
        
        result_state = workflow_with_mocks._save_conversation_node(sample_state)
        
        # Verify error was appended (doesn't fail the workflow)
        assert "Conversation save error: Save failed" in result_state["error"]
    
    @patch('app.chat_workflow.conversation_manager')
    def test_process_chat_query_success(self, mock_conversation_manager, workflow_with_mocks):
        """Test successful end-to-end chat query processing."""
        # Mock conversation history
        mock_conversation_manager.format_conversation_for_prompt.return_value = "Formatted history"
        
        # Mock workflow invoke to return a complete final state
        mock_final_state = {
            "user_id": "test_user",
            "query": "What smartphones do you have?",
            "response": "Here are our available smartphones...",
            "conversation_context": {"is_followup_question": False, "context_used": True},
            "original_query": "What smartphones do you have?",
            "optimized_queries": ["smartphone options", "phone models"],
            "llm_reasoning": "Generated multiple queries for comprehensive search",
            "context_usage": "Used conversation context to understand user needs",
            "retrieved_documents": [{"content": "iPhone info"}, {"content": "Samsung info"}],
            "retrieval_metadata": {"top_k": 3, "total_results": 2},
            "generation_metadata": {"model": "claude-3-haiku-20240307", "tokens": 200},
            "error": ""
        }
        
        # Mock the entire workflow object to avoid Pydantic v1 issues
        mock_workflow = Mock()
        mock_workflow.invoke.return_value = mock_final_state
        workflow_with_mocks.workflow = mock_workflow
        
        result = workflow_with_mocks.process_chat_query("test_user", "What smartphones do you have?")
        
        # Verify conversation history was formatted
        mock_conversation_manager.format_conversation_for_prompt.assert_called_once_with("test_user")
        
        # Verify workflow was invoked with correct initial state
        mock_workflow.invoke.assert_called_once()
        initial_state = mock_workflow.invoke.call_args[0][0]
        assert initial_state["user_id"] == "test_user"
        assert initial_state["query"] == "What smartphones do you have?"
        assert initial_state["conversation_history"] == "Formatted history"
        
        # Verify response structure
        assert result["user_id"] == "test_user"
        assert result["query"] == "What smartphones do you have?"
        assert result["response"] == "Here are our available smartphones..."
        assert result["conversation_context"]["is_followup_question"] is False
        assert result["search_enhancement"]["optimized_queries"] == ["smartphone options", "phone models"]
        assert result["metadata"]["documents_used"] == 2
        assert result["metadata"]["has_error"] is False
        assert result["metadata"]["conversation_aware"] is True
        assert result["debug_info"] is None  # No error, so no debug info
    
    @patch('app.chat_workflow.conversation_manager')
    def test_process_chat_query_with_workflow_error(self, mock_conversation_manager, workflow_with_mocks):
        """Test chat query processing when workflow raises an exception."""
        # Mock conversation history
        mock_conversation_manager.format_conversation_for_prompt.return_value = "Formatted history"
        
        # Mock the entire workflow object to raise an exception
        mock_workflow = Mock()
        mock_workflow.invoke.side_effect = Exception("Workflow failed")
        workflow_with_mocks.workflow = mock_workflow
        
        result = workflow_with_mocks.process_chat_query("test_user", "test query")
        
        # Verify error response structure
        assert result["user_id"] == "test_user"
        assert result["query"] == "test query"
        assert "I apologize, but I encountered a system error" in result["response"]
        assert result["conversation_context"]["is_followup_question"] is False
        assert result["search_enhancement"]["reasoning"] == "Workflow error occurred"
        assert result["metadata"]["has_error"] is True
        assert result["metadata"]["documents_used"] == 0
        assert result["debug_info"]["error"] == "Workflow error: Workflow failed"
        assert result["debug_info"]["conversation_history"] == "Formatted history"
    
    @patch('app.chat_workflow.conversation_manager')
    def test_process_chat_query_with_state_error(self, mock_conversation_manager, workflow_with_mocks):
        """Test chat query processing when final state has errors."""
        # Mock conversation history
        mock_conversation_manager.format_conversation_for_prompt.return_value = "History"
        
        # Mock workflow invoke to return state with error
        mock_final_state = {
            "user_id": "test_user",
            "query": "test query",
            "response": "Error response",
            "conversation_context": {"is_followup_question": False},
            "original_query": "test query",
            "optimized_queries": [],
            "llm_reasoning": "Error occurred",
            "context_usage": "Error",
            "retrieved_documents": [],
            "retrieval_metadata": {"error": True},
            "generation_metadata": {"error": True},
            "error": "Something went wrong in retrieval"
        }
        
        # Mock the entire workflow object to return state with error
        mock_workflow = Mock()
        mock_workflow.invoke.return_value = mock_final_state
        workflow_with_mocks.workflow = mock_workflow
        
        result = workflow_with_mocks.process_chat_query("test_user", "test query")
        
        # Verify error is handled properly
        assert result["metadata"]["has_error"] is True
        assert result["metadata"]["documents_used"] == 0
        assert result["debug_info"]["error"] == "Something went wrong in retrieval"
        assert result["debug_info"] is not None  # Debug info should be included when there's an error


class TestChatWorkflowIntegration:
    """Integration tests for the chat workflow."""
    
    @patch('app.chat_workflow.conversation_manager')
    def test_minimal_integration_with_real_agents(self, mock_conversation_manager, mock_vector_store):
        """Test workflow with real agent classes but mocked external dependencies."""
        mock_conversation_manager.format_conversation_for_prompt.return_value = ""
        
        with patch('app.agents.retriever_agent.anthropic.Anthropic'), \
             patch('app.agents.responder_agent.anthropic.Anthropic'):
            
            workflow = ChatRAGWorkflow(mock_vector_store)
            
            # Verify agents are real instances
            assert isinstance(workflow.retriever_agent, RetrieverAgent)
            assert isinstance(workflow.responder_agent, ResponderAgent)
            assert workflow.workflow is not None
    
    def test_state_transitions(self, workflow_with_mocks, sample_state):
        """Test that state is properly passed between workflow nodes."""
        # Test retrieve -> respond transition
        after_retrieve = workflow_with_mocks._retrieve_node(sample_state)
        assert after_retrieve["retrieved_documents"] is not None
        assert after_retrieve["retrieval_metadata"] is not None
        
        # Test respond node uses retrieve results
        after_respond = workflow_with_mocks._respond_node(after_retrieve)
        assert after_respond["response"] != ""
        assert after_respond["conversation_context"] is not None
        
        # Test save conversation uses both results
        final_state = workflow_with_mocks._save_conversation_node(after_respond)
        assert final_state == after_respond  # Should return same state