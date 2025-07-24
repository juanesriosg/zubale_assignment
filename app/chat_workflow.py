"""
Enhanced RAG Workflow with conversation history support.
"""

from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from app.agents.retriever_agent import RetrieverAgent
from app.agents.responder_agent import ResponderAgent
from app.conversation import conversation_manager
from app.vector_store import VectorStore


class ChatWorkflowState(TypedDict):
    user_id: str
    query: str
    conversation_history: str
    original_query: str
    optimized_queries: list
    llm_reasoning: str
    context_usage: str
    retrieved_documents: list
    retrieval_metadata: dict
    response: str
    conversation_context: dict
    retrieved_context: str
    search_reasoning: str
    generation_metadata: dict
    error: str


class ChatRAGWorkflow:
    """
    Enhanced RAG Workflow that maintains conversation context and uses
    LLM-powered agents for both retrieval and response generation.
    """
    
    def __init__(self, vector_store: VectorStore):
        self.retriever_agent = RetrieverAgent(vector_store)
        self.responder_agent = ResponderAgent()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        workflow = StateGraph(ChatWorkflowState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("respond", self._respond_node)
        workflow.add_node("save_conversation", self._save_conversation_node)
        
        # Add edges
        workflow.add_edge("retrieve", "respond")
        workflow.add_edge("respond", "save_conversation")
        workflow.add_edge("save_conversation", END)
        
        # Set entry point
        workflow.set_entry_point("retrieve")
        
        return workflow.compile()
    
    def _retrieve_node(self, state: ChatWorkflowState) -> ChatWorkflowState:
        """Node for enhanced document retrieval with conversation context."""
        try:
            result = self.retriever_agent.retrieve(
                current_query=state["query"],
                conversation_history=state["conversation_history"]
            )
            
            state["original_query"] = result["original_query"]
            state["optimized_queries"] = result["optimized_queries"]
            state["llm_reasoning"] = result["llm_reasoning"]
            state["context_usage"] = result["context_usage"]
            state["retrieved_documents"] = result["retrieved_documents"]
            state["retrieval_metadata"] = result["retrieval_metadata"]
            
        except Exception as e:
            state["error"] = f"Retrieval error: {str(e)}"
            state["retrieved_documents"] = []
            state["retrieval_metadata"] = {"error": True}
            state["llm_reasoning"] = "Error in retrieval processing"
            state["context_usage"] = "Error"
        
        return state
    
    def _respond_node(self, state: ChatWorkflowState) -> ChatWorkflowState:
        """Node for contextually aware response generation."""
        try:
            if state.get("error") and not state["retrieved_documents"]:
                state["response"] = "I apologize, but I encountered an issue retrieving relevant product information. Please try rephrasing your question or contact support."
                state["generation_metadata"] = {"error": True}
                state["conversation_context"] = {"is_followup_question": False, "context_used": False}
                return state
            
            result = self.responder_agent.generate_response(
                current_query=state["query"],
                retrieved_docs=state["retrieved_documents"],
                conversation_history=state["conversation_history"],
                search_reasoning=state.get("llm_reasoning", "")
            )
            
            state["response"] = result["response"]
            state["conversation_context"] = result["conversation_context"]
            state["retrieved_context"] = result["retrieved_context"]
            state["search_reasoning"] = result["search_reasoning"]
            state["generation_metadata"] = result["generation_metadata"]
            
            if "error" in result:
                state["error"] = result["error"]
                
        except Exception as e:
            state["error"] = f"Response generation error: {str(e)}"
            state["response"] = "I apologize, but I encountered an issue generating a response. Please try again or contact support."
            state["generation_metadata"] = {"error": True}
            state["conversation_context"] = {"is_followup_question": False, "context_used": False}
        
        return state
    
    def _save_conversation_node(self, state: ChatWorkflowState) -> ChatWorkflowState:
        """Node to save the conversation to history."""
        try:
            # Save user message
            conversation_manager.add_message(
                user_id=state["user_id"],
                role="user",
                content=state["query"],
                metadata={
                    "is_followup": state.get("conversation_context", {}).get("is_followup_question", False),
                    "optimized_queries": state.get("optimized_queries", [])
                }
            )
            
            # Save assistant response
            conversation_manager.add_message(
                user_id=state["user_id"],
                role="assistant",
                content=state["response"],
                metadata={
                    "documents_used": len(state.get("retrieved_documents", [])),
                    "search_reasoning": state.get("llm_reasoning", ""),
                    "generation_metadata": state.get("generation_metadata", {})
                }
            )
            
        except Exception as e:
            # Don't fail the whole workflow if conversation saving fails
            state["error"] = state.get("error", "") + f" | Conversation save error: {str(e)}"
        
        return state
    
    def process_chat_query(self, user_id: str, query: str) -> Dict[str, Any]:
        """Process a chat query with full conversation context."""
        
        # Get conversation history
        conversation_history = conversation_manager.format_conversation_for_prompt(user_id)
        
        initial_state = ChatWorkflowState(
            user_id=user_id,
            query=query,
            conversation_history=conversation_history,
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
        
        try:
            final_state = self.workflow.invoke(initial_state)
            
            return {
                "user_id": final_state["user_id"],
                "query": final_state["query"],
                "response": final_state["response"],
                "conversation_context": final_state.get("conversation_context", {}),
                "search_enhancement": {
                    "original_query": final_state.get("original_query", query),
                    "optimized_queries": final_state.get("optimized_queries", []),
                    "reasoning": final_state.get("llm_reasoning", ""),
                    "context_usage": final_state.get("context_usage", "")
                },
                "metadata": {
                    "retrieval": final_state.get("retrieval_metadata", {}),
                    "generation": final_state.get("generation_metadata", {}),
                    "documents_used": len(final_state.get("retrieved_documents", [])),
                    "has_error": bool(final_state.get("error")),
                    "conversation_aware": True
                },
                "debug_info": {
                    "retrieved_documents": final_state.get("retrieved_documents", []),
                    "retrieved_context": final_state.get("retrieved_context", ""),
                    "search_reasoning": final_state.get("search_reasoning", ""),
                    "conversation_history": conversation_history,
                    "error": final_state.get("error", "")
                } if final_state.get("error") else None
            }
            
        except Exception as e:
            return {
                "user_id": user_id,
                "query": query,
                "response": "I apologize, but I encountered a system error while processing your request. Please try again later or contact support.",
                "conversation_context": {"is_followup_question": False, "context_used": False},
                "search_enhancement": {
                    "original_query": query,
                    "optimized_queries": [],
                    "reasoning": "Workflow error occurred",
                    "context_usage": "Error"
                },
                "metadata": {
                    "retrieval": {"error": True},
                    "generation": {"error": True},
                    "documents_used": 0,
                    "has_error": True,
                    "conversation_aware": True
                },
                "debug_info": {
                    "error": f"Workflow error: {str(e)}",
                    "conversation_history": conversation_history
                }
            }