"""
Langfuse integration for observability and tracing.
"""
from typing import Optional
from functools import wraps
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from src.logger import logger


_langfuse_client: Optional[Langfuse] = None
_langfuse_enabled: bool = False


def initialize_langfuse(host: str, public_key: str, secret_key: str) -> None:
    """
    Initialize the Langfuse client.
    
    Args:
        host: Langfuse host URL (e.g., http://localhost:3000)
        public_key: Langfuse public key
        secret_key: Langfuse secret key
    """
    global _langfuse_client, _langfuse_enabled
    
    try:
        _langfuse_client = Langfuse(
            host=host,
            public_key=public_key,
            secret_key=secret_key,
        )
        _langfuse_enabled = True
        logger.info(f"Langfuse initialized successfully with host: {host}")
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse: {e}")
        _langfuse_enabled = False


def is_langfuse_enabled() -> bool:
    """Check if Langfuse is enabled and initialized."""
    return _langfuse_enabled


def get_langfuse_client() -> Optional[Langfuse]:
    """Get the Langfuse client instance."""
    return _langfuse_client


def flush_langfuse() -> None:
    """Flush any pending traces to Langfuse."""
    if _langfuse_client:
        try:
            _langfuse_client.flush()
            logger.debug("Langfuse traces flushed")
        except Exception as e:
            logger.error(f"Failed to flush Langfuse traces: {e}")


def conditional_observe(func):
    """
    Decorator that applies @observe only if Langfuse is enabled.
    If disabled, the function executes normally without tracing.
    """
    if _langfuse_enabled:
        return observe()(func)
    return func
