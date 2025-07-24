import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.models import (
    HealthResponse, ChatQueryRequest, ChatQueryResponse, ConversationHistoryResponse
)
from app.vector_store import initialize_vector_store, VectorStore
from app.chat_workflow import ChatRAGWorkflow
from app.conversation import conversation_manager
from app.config import settings

# Configure logging
logging.basicConfig(level=getattr(logging, settings.log_level.upper()))
logger = logging.getLogger(__name__)

# Global variables for dependency injection
vector_store: VectorStore = None
chat_rag_workflow: ChatRAGWorkflow = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global vector_store, chat_rag_workflow
    
    logger.info("Initializing chat application...")
    
    try:
        # Initialize vector store
        vector_store = initialize_vector_store()
        logger.info("Vector store initialized successfully")
        
        # Initialize chat workflow
        chat_rag_workflow = ChatRAGWorkflow(vector_store)
        logger.info("Chat RAG workflow initialized successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    
    # Shutdown
    logger.info("Shutting down chat application...")


app = FastAPI(
    title="Product Chat Bot",
    description="A RAG-based chat system for product conversations with history",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


def get_chat_rag_workflow() -> ChatRAGWorkflow:
    if chat_rag_workflow is None:
        raise HTTPException(
            status_code=503, 
            detail="Chat RAG workflow not initialized. Please check server logs."
        )
    return chat_rag_workflow


def get_vector_store() -> VectorStore:
    if vector_store is None:
        raise HTTPException(
            status_code=503, 
            detail="Vector store not initialized. Please check server logs."
        )
    return vector_store


@app.get("/")
async def serve_web_app():
    """Serve the web application."""
    return FileResponse("static/index.html")

@app.get("/api", response_model=dict)
async def api_info():
    return {
        "message": "Product Chat Bot API",
        "version": "2.0.0",
        "endpoints": {
            "chat": "/chat",
            "conversation": "/conversation/{user_id}",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(
    vector_store: VectorStore = Depends(get_vector_store)
):
    try:
        indexed_docs = len(vector_store.documents)
        store_status = "healthy" if indexed_docs > 0 else "empty"
        
        return HealthResponse(
            status="healthy",
            version="2.0.0",
            vector_store_status=store_status,
            indexed_documents=indexed_docs
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version="2.0.0",
            vector_store_status="error",
            indexed_documents=0
        )


@app.post("/chat", response_model=ChatQueryResponse)
async def process_chat_query(
    request: ChatQueryRequest,
    workflow: ChatRAGWorkflow = Depends(get_chat_rag_workflow)
):
    """
    Process a chat query with conversation history support.
    This endpoint maintains conversation context and provides enhanced responses.
    """
    try:
        logger.info(f"Processing chat query from user {request.user_id}: {request.query}")
        
        # Process the query through the enhanced chat RAG workflow
        result = workflow.process_chat_query(request.user_id, request.query)
        
        logger.info(f"Chat query processed successfully for user {request.user_id}")
        
        return ChatQueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Error processing chat query for user {request.user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while processing chat query: {str(e)}"
        )


@app.get("/conversation/{user_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(user_id: str, limit: int = 20):
    """
    Get conversation history for a specific user.
    """
    try:
        logger.info(f"Retrieving conversation history for user {user_id}")
        
        # Get recent conversation history
        messages = conversation_manager.get_recent_history(user_id, max_messages=limit)
        
        # Format messages for response
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "metadata": msg.metadata or {}
            })
        
        return ConversationHistoryResponse(
            user_id=user_id,
            messages=formatted_messages,
            total_messages=len(formatted_messages)
        )
        
    except Exception as e:
        logger.error(f"Error retrieving conversation history for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while retrieving conversation history: {str(e)}"
        )


@app.delete("/conversation/{user_id}")
async def clear_conversation_history(user_id: str):
    """
    Clear conversation history for a specific user.
    """
    try:
        logger.info(f"Clearing conversation history for user {user_id}")
        
        conversation_manager.clear_conversation(user_id)
        
        return {
            "message": f"Conversation history cleared for user {user_id}",
            "user_id": user_id,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Error clearing conversation history for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error while clearing conversation history: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)