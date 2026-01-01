from langchain_chroma import Chroma
from app.core.config import get_settings
import os

settings = get_settings()

def create_chroma(docs, embeddings):
    return Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=settings.COLLECTION_NAME,
        persist_directory=settings.CHROMA_PERSIST_DIR
    )

def get_chroma(embeddings):
    if not os.path.exists(settings.CHROMA_PERSIST_DIR):
        return None
        
    return Chroma(
        embedding_function=embeddings,
        collection_name=settings.COLLECTION_NAME,
        persist_directory=settings.CHROMA_PERSIST_DIR
    )
