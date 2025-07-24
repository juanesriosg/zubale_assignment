"""
Responder Agent - Enhanced with conversation history management.
"""

from typing import Dict, Any, List
import anthropic
from app.config import settings
from app.agents.prompts import RESPONDER_AGENT_PROMPT


class ResponderAgent:
    """
    Enhanced Responder Agent that maintains conversation context
    and generates contextually aware responses.
    """
    
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.name = "responder_agent"
    
    def generate_response(
        self, 
        current_query: str, 
        retrieved_docs: List[Dict[str, Any]], 
        conversation_history: str = "",
        search_reasoning: str = ""
    ) -> Dict[str, Any]:
        """
        Generate a contextually aware response using conversation history.
        """
        
        # Create context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            search_info = ""
            if "search_query_used" in doc:
                search_info = f" (Found via: '{doc['search_query_used']}')"
            
            context_parts.append(
                f"Document {i} (Relevance: {doc['score']:.3f}){search_info}:\n{doc['content']}"
            )
        
        retrieved_context = "\n\n".join(context_parts) if context_parts else "No relevant documents found."
        
        # Format the conversation history
        history_text = conversation_history if conversation_history.strip() else "This is the start of the conversation."
        
        # Create the prompt with conversation context
        prompt = RESPONDER_AGENT_PROMPT.format(
            conversation_history=history_text,
            current_query=current_query,
            retrieved_context=retrieved_context,
            search_reasoning=search_reasoning
        )
        
        try:
            response = self.client.messages.create(
                model=settings.llm_model,
                max_tokens=1000,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text
            
            # Analyze if this appears to be a follow-up question
            is_followup = self._detect_followup_question(current_query, conversation_history)
            
            return {
                "query": current_query,
                "response": response_text,
                "conversation_context": {
                    "is_followup_question": is_followup,
                    "conversation_length": len(conversation_history.split('\n')) if conversation_history else 0,
                    "context_used": bool(conversation_history.strip())
                },
                "retrieved_context": retrieved_context,
                "search_reasoning": search_reasoning,
                "generation_metadata": {
                    "model": settings.llm_model,
                    "documents_used": len(retrieved_docs),
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "conversation_aware": True
                }
            }
            
        except Exception as e:
            return {
                "query": current_query,
                "response": f"I apologize, but I encountered an error while generating a response: {str(e)}. Please try again or contact support.",
                "conversation_context": {
                    "is_followup_question": False,
                    "conversation_length": 0,
                    "context_used": False
                },
                "retrieved_context": retrieved_context,
                "search_reasoning": search_reasoning,
                "error": str(e),
                "generation_metadata": {
                    "model": settings.llm_model,
                    "documents_used": len(retrieved_docs),
                    "error": True,
                    "conversation_aware": True
                }
            }
    
    def _detect_followup_question(self, current_query: str, conversation_history: str) -> bool:
        """
        Simple heuristic to detect if the current query is a follow-up question.
        """
        if not conversation_history.strip():
            return False
        
        # Check for common follow-up indicators
        followup_indicators = [
            "what about", "how about", "and what", "also", "too",
            "compared to", "vs", "versus", "difference",
            "that one", "this one", "it", "they", "those"
        ]
        
        query_lower = current_query.lower()
        
        # Check if query starts with common follow-up phrases
        followup_starters = ["what about", "how about", "and", "also", "what's the"]
        if any(query_lower.startswith(starter) for starter in followup_starters):
            return True
        
        # Check for pronouns and references
        if any(indicator in query_lower for indicator in followup_indicators):
            return True
        
        # Check if query is much shorter than typical initial queries (likely a follow-up)
        if len(current_query.split()) <= 4 and len(conversation_history) > 100:
            return True
        
        return False