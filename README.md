# Product Chat Bot - RAG Pipeline with Conversation History

A conversational AI system that processes product questions using a Retrieval-Augmented Generation (RAG) pipeline with multi-agent architecture and conversation memory.

## Architecture

- **FastAPI** microservice with chat endpoints and web interface
- **Enhanced Multi-agent system** using LangGraph:
  - **Retriever Agent**: LLM-powered query optimization with conversation context
  - **Responder Agent**: Context-aware response generation with chat history
- **Conversation Management**: Per-user chat history with follow-up detection
- **Vector Store**: FAISS-based storage with sentence-transformers embeddings
- **LLM Integration**: Anthropic Claude for both agents and response generation

## Features

- **Conversational Chat Interface** - Web UI and REST API
- **Conversation History** - Per-user chat memory and context
- **Follow-up Question Detection** - Intelligent context awareness
- **LLM-Enhanced Retrieval** - Query optimization using Claude
- **Multi-agent Architecture** - Specialized agents for retrieval and response
- **RAG Implementation** - Semantic search with FAISS
- **Product Knowledge Base** - 8 sample products (smartphones, laptops, etc.)
- **Input Validation** - Comprehensive error handling
- **Environment Configuration** - Flexible deployment options
- **Testing Suite** - Unit tests for chat functionality
- **Docker Support** - Full containerization
- **Health Monitoring** - System status endpoints

## Quick Start

### Prerequisites

- Python 3.11+
- Anthropic API key
- Docker (optional)

### Installation

1. **Clone and setup:**

   ```bash
   cd test_zubale
   pip install -r requirements.txt
   ```

2. **Configure environment:**

   ```bash
   cp .env.example .env
   # Edit .env and add your Anthropic API key
   ```

3. **Run the service:**

   ```bash
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
   ```

4. **Open the web chat interface:**

   ```
   Open your browser to: http://localhost:8001/
   ```

5. **Or test the API directly:**

   ```bash
   curl -X POST "http://localhost:8001/chat" \
        -H "Content-Type: application/json" \
        -d '{"user_id": "user_123", "query": "I need a smartphone for photography"}'
   ```

### Docker Setup

1. **Build and run:**

   ```bash
   docker-compose up --build
   ```

2. **Test:**

   ```bash
   curl http://localhost:8001/health
   ```

## API Endpoints

### POST /chat

Process conversational product queries with history.

**Request:**

```json
{
  "user_id": "string",
  "query": "string"
}
```

**Response:**

```json
{
  "user_id": "string",
  "query": "string", 
  "response": "string",
  "conversation_context": {
    "is_followup_question": false,
    "conversation_length": 2,
    "context_used": true
  },
  "search_enhancement": {
    "original_query": "string",
    "optimized_queries": ["query1", "query2"],
    "reasoning": "explanation",
    "context_usage": "how context was used"
  },
  "metadata": {
    "retrieval": {"top_k": 3, "total_results": 2},
    "generation": {"model": "claude-3-haiku-20240307", "documents_used": 2},
    "documents_used": 2,
    "has_error": false,
    "conversation_aware": true
  }
}
```

### GET /conversation/{user_id}

Get conversation history for a user.

**Response:**

```json
{
  "user_id": "string",
  "messages": [
    {
      "role": "user|assistant",
      "content": "string",
      "timestamp": "2025-01-01T10:00:00",
      "metadata": {}
    }
  ],
  "total_messages": 4
}
```

### DELETE /conversation/{user_id}

Clear conversation history for a user.

### GET /health

Service health check.

**Response:**

```json
{
  "status": "healthy",
  "version": "2.0.0",
  "vector_store_status": "healthy",
  "indexed_documents": 8
}
```

## Configuration

Environment variables (see `.env.example`):

- `ANTHROPIC_API_KEY`: Anthropic API key
- `TOP_K_RETRIEVAL`: Number of documents to retrieve (default: 3)
- `LLM_MODEL`: Claude model name (default: claude-3-haiku-20240307)
- `EMBEDDING_MODEL`: Sentence transformer model (default: all-MiniLM-L6-v2)
- `LOG_LEVEL`: Logging level (default: INFO)

## Testing

Run the test suite:

```bash
# All tests
pytest

# Specific test modules
pytest tests/test_vector_store.py
pytest tests/test_agents.py
pytest tests/test_chat_system.py
pytest tests/test_chat_api.py
pytest tests/test_chat_workflow.py
pytest tests/test_config.py
pytest tests/test_integration.py
pytest tests/test_models.py

# With coverage
pytest --cov=app
```

## Project Structure

```
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI application with chat endpoints
│   ├── config.py         # Configuration settings
│   ├── models.py         # Pydantic models for API
│   ├── conversation.py   # Conversation history management
│   ├── vector_store.py   # FAISS vector store implementation
│   ├── chat_workflow.py  # Enhanced chat workflow with history
│   └── agents/           # Enhanced agent implementations
│       ├── __init__.py
│       ├── retriever_agent.py    # LLM-enhanced query optimization
│       ├── responder_agent.py    # Context-aware response generation
│       └── prompts.py            # Specialized prompt templates
├── static/
│   └── index.html        # Web chat interface
├── data/                 # Product documents corpus (8 products)
│   ├── product_1.txt     # iPhone 15 Pro Max
│   ├── product_2.txt     # MacBook Air M2
│   └── ...               # Smartphones, laptops, headphones, gaming
├── tests/                # Unit tests
│   ├── test_vector_store.py
│   ├── test_agents.py         # Tests for agent functionality
│   ├── test_chat_system.py    # Tests for chat functionality
│   └── test_chat_api.py       # Tests for API endpoints
├── vector_store/         # Generated FAISS indices
├── Dockerfile
├── docker-compose.yml
├── requirements.txt      # Cleaned up dependencies
└── README.md
```

## Sample Conversations

### **Initial Questions:**

- "I need a smartphone for photography"
- "Show me gaming laptops under $2000"
- "What headphones have the best noise cancellation?"

### **Follow-up Questions:**

- "What about battery life?" (after discussing smartphones)
- "How much do they cost?" (following product recommendations)
- "Compare them" (after hearing about multiple options)

### **Multi-turn Conversations:**

```
User: "I'm looking for a new laptop"
Bot: [Recommends MacBook Air M2 and Dell XPS 13 Plus]

User: "What about battery life?"
Bot: [Compares battery specs, referencing previous laptop discussion]

User: "Which one is better for programming?"
Bot: [Contextual comparison focusing on programming features]
```

## Implementation Notes

### RAG Pipeline

- Uses `sentence-transformers` for embeddings (all-MiniLM-L6-v2)
- FAISS for efficient similarity search with cosine similarity
- Context-aware prompt engineering with Anthropic Claude to reduce hallucination

### Enhanced Multi-Agent Design

- **Retriever Agent**: LLM-powered query optimization with conversation context awareness
- **Responder Agent**: Context-aware response generation maintaining chat history
- **LangGraph Workflow**: Orchestrates conversational agent communication with history management
- **Conversation Manager**: Per-user memory with follow-up question detection

### Production Considerations

- Environment-based configuration
- Comprehensive error handling and logging
- Input validation with Pydantic
- Health checks for monitoring
- Docker containerization for deployment
- Unit tests covering core functionality

## Extensions

Potential enhancements:

- Persistent vector store (PostgreSQL with pgvector)
- Advanced retrieval (hybrid search, re-ranking)
- Streaming responses
- Rate limiting and authentication
- Metrics and observability
- Multi-language support
