import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    top_k_retrieval: int = os.getenv("TOP_K_RETRIEVAL", 3)
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    llm_model: str = os.getenv("LLM_MODEL", "claude-3-haiku-20240307")
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "./vector_store")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        env_file = ".env"


settings = Settings()