from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # App
    APP_NAME: str = "Speculative RAG API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # LLM - Groq
    GROQ_API_KEY: str
    DRAFTER_MODEL: str = "llama-3.1-8b-instant"
    VERIFIER_MODEL: str = "llama-3.3-70b-versatile"
    
    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Vector Store
    CHROMA_PERSIST_DIR: str = "data/chroma"
    COLLECTION_NAME: str = "rag_collection"
    
    # RAG
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Langfuse
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    class Config:
        env_file = ".env"
        extra = "ignore"

@lru_cache()
def get_settings():
    return Settings()
