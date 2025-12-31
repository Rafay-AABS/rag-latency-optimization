from langchain_groq import ChatGroq

def get_drafter():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

def get_verifier():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )
