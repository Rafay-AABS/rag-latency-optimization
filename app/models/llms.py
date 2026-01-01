from langchain_groq import ChatGroq
from app.core.config import get_settings

settings = get_settings()

def get_drafter():
    return ChatGroq(
        model=settings.DRAFTER_MODEL,
        temperature=0,
        api_key=settings.GROQ_API_KEY
    )

def get_verifier():
    return ChatGroq(
        model=settings.VERIFIER_MODEL,
        temperature=0,
        api_key=settings.GROQ_API_KEY
    )
