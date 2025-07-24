"""
Conversation history management for the chat system.
"""

from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = datetime.now()
    metadata: Optional[Dict] = None


class ConversationHistory(BaseModel):
    user_id: str
    messages: List[ChatMessage] = []
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()


class ConversationManager:
    """In-memory conversation storage. In production, this would use a database."""
    
    def __init__(self):
        self.conversations: Dict[str, ConversationHistory] = {}
    
    def get_conversation(self, user_id: str) -> ConversationHistory:
        """Get conversation history for a user."""
        if user_id not in self.conversations:
            self.conversations[user_id] = ConversationHistory(user_id=user_id)
        return self.conversations[user_id]
    
    def add_message(self, user_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        """Add a message to the conversation history."""
        conversation = self.get_conversation(user_id)
        message = ChatMessage(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        conversation.messages.append(message)
        conversation.updated_at = datetime.now()
    
    def get_recent_history(self, user_id: str, max_messages: int = 10) -> List[ChatMessage]:
        """Get recent conversation history for context."""
        conversation = self.get_conversation(user_id)
        return conversation.messages[-max_messages:] if conversation.messages else []
    
    def format_conversation_for_prompt(self, user_id: str, max_messages: int = 6) -> str:
        """Format conversation history for use in prompts."""
        messages = self.get_recent_history(user_id, max_messages)
        
        if not messages:
            return "No previous conversation history."
        
        formatted_messages = []
        for msg in messages:
            timestamp = msg.timestamp.strftime("%H:%M")
            formatted_messages.append(f"[{timestamp}] {msg.role.title()}: {msg.content}")
        
        return "\n".join(formatted_messages)
    
    def clear_conversation(self, user_id: str):
        """Clear conversation history for a user."""
        if user_id in self.conversations:
            del self.conversations[user_id]
    
    def get_conversation_summary(self, user_id: str) -> str:
        """Get a summary of the conversation for context."""
        messages = self.get_recent_history(user_id, max_messages=20)
        
        if not messages:
            return "No conversation history available."
        
        # Simple summary - in production, you might use an LLM for this
        user_messages = [msg.content for msg in messages if msg.role == "user"]
        if user_messages:
            return f"User has asked about: {', '.join(user_messages[-3:])}"
        
        return "No user questions in recent history."


# Global conversation manager instance
conversation_manager = ConversationManager()