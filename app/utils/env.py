import os
from dotenv import load_dotenv

def load_env():
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Missing GROQ_API_KEY in .env")
