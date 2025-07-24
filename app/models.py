from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class ChatQueryRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Unique identifier for the user")
    query: str = Field(..., min_length=1, max_length=1000, description="User's product question or follow-up")


class ChatQueryResponse(BaseModel):
    user_id: str
    query: str
    response: str
    conversation_context: Dict[str, Any]
    search_enhancement: Dict[str, Any]
    metadata: Dict[str, Any]
    debug_info: Optional[Dict[str, Any]] = None


class ConversationHistoryResponse(BaseModel):
    user_id: str
    messages: List[Dict[str, Any]]
    total_messages: int


class HealthResponse(BaseModel):
    status: str
    version: str
    vector_store_status: str
    indexed_documents: int