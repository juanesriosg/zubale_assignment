"""
Retriever Agent - Enhanced with LLM for intelligent query processing.
"""

import json
from typing import Dict, Any, List
import anthropic
from app.vector_store import VectorStore
from app.config import settings
from app.agents.prompts import RETRIEVER_AGENT_PROMPT


class RetrieverAgent:
    """
    Enhanced Retriever Agent that uses LLM to optimize search queries
    based on conversation context and user intent.
    """
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.name = "retriever_agent"
    
    def _optimize_query_with_llm(self, current_query: str, conversation_history: str) -> Dict[str, Any]:
        """Use LLM to optimize the search query based on conversation context."""
        
        prompt = RETRIEVER_AGENT_PROMPT.format(
            conversation_history=conversation_history,
            current_query=current_query
        )
        
        try:
            response = self.client.messages.create(
                model=settings.llm_model,
                max_tokens=500,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse the JSON response
            response_text = response.content[0].text.strip()
            
            # Extract JSON from the response (handle potential markdown formatting)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_text = response_text[json_start:json_end]
            else:
                # Fallback if no proper JSON found
                return {
                    "search_queries": [current_query],
                    "reasoning": "Could not parse LLM response, using original query",
                    "context_used": "None"
                }
            
            parsed_response = json.loads(json_text)
            
            # Validate the response structure
            if not isinstance(parsed_response.get("search_queries"), list):
                raise ValueError("Invalid search_queries format")
            
            return parsed_response
            
        except Exception as e:
            # Fallback to original query if LLM processing fails
            return {
                "search_queries": [current_query],
                "reasoning": f"LLM processing failed: {str(e)}, using original query",
                "context_used": "Error in context processing",
                "error": str(e)
            }
    
    def retrieve(self, current_query: str, conversation_history: str = "", top_k: int = None) -> Dict[str, Any]:
        """
        Enhanced retrieval that uses LLM to optimize queries based on conversation context.
        """
        if top_k is None:
            top_k = settings.top_k_retrieval
        
        # Use LLM to optimize the search query
        llm_optimization = self._optimize_query_with_llm(current_query, conversation_history)
        
        # Perform searches with the optimized queries
        all_results = []
        search_metadata = []
        
        for search_query in llm_optimization["search_queries"]:
            results = self.vector_store.search(search_query, top_k=top_k)
            
            # Add search query info to each result
            for result in results:
                result["search_query_used"] = search_query
            
            all_results.extend(results)
            search_metadata.append({
                "query": search_query,
                "results_count": len(results)
            })
        
        # Remove duplicates based on content while preserving the best scores
        seen_content = {}
        unique_results = []
        
        for result in all_results:
            content = result["content"]
            if content not in seen_content or result["score"] > seen_content[content]["score"]:
                seen_content[content] = result
        
        unique_results = list(seen_content.values())
        
        # Sort by score and limit to top_k
        unique_results.sort(key=lambda x: x["score"], reverse=True)
        final_results = unique_results[:top_k]
        
        return {
            "original_query": current_query,
            "optimized_queries": llm_optimization["search_queries"],
            "llm_reasoning": llm_optimization["reasoning"],
            "context_usage": llm_optimization["context_used"],
            "retrieved_documents": final_results,
            "retrieval_metadata": {
                "top_k": top_k,
                "total_unique_results": len(unique_results),
                "final_results_count": len(final_results),
                "search_metadata": search_metadata,
                "llm_error": llm_optimization.get("error")
            }
        }