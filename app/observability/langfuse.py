from langfuse.langchain import CallbackHandler

def get_langfuse_handler():
    try:
        return CallbackHandler()
    except Exception:
        return None
