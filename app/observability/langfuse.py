import logging
import os
from langfuse.langchain import CallbackHandler
from app.core.config import get_settings

logger = logging.getLogger(__name__)

def get_langfuse_handler():
    settings = get_settings()
    try:
        logger.info(f"Initializing Langfuse with Host: {settings.LANGFUSE_HOST}")
        
        # Set environment variables for Langfuse to pick up automatically
        # This avoids issues with varying constructor arguments across versions
        if settings.LANGFUSE_PUBLIC_KEY:
            os.environ["LANGFUSE_PUBLIC_KEY"] = settings.LANGFUSE_PUBLIC_KEY
        if settings.LANGFUSE_SECRET_KEY:
            os.environ["LANGFUSE_SECRET_KEY"] = settings.LANGFUSE_SECRET_KEY
        if settings.LANGFUSE_HOST:
            os.environ["LANGFUSE_HOST"] = settings.LANGFUSE_HOST

        # Initialize without arguments, letting it read from env vars
        handler = CallbackHandler()
        
        # Test auth
        if hasattr(handler, 'auth_check'):
            if not handler.auth_check():
                logger.error("Langfuse authentication failed.")
                return None
                
        logger.info("Langfuse handler initialized successfully.")
        return handler
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse handler: {e}")
        return None
