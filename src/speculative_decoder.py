from src.logger import logger
from langfuse.decorators import observe, langfuse_context


@observe()
def speculative_decode(prompt: str, draft_model, target_model) -> str:
    """Execute speculative decoding strategy."""
    logger.debug("Starting speculative decoding")
    
    # Update trace with input information
    langfuse_context.update_current_observation(
        input={"prompt_length": len(prompt)},
        metadata={"strategy": "speculative_decoding"}
    )
    
    draft_text = draft_model.generate(prompt)
    final_text = target_model.verify(prompt, draft_text)
    
    # Add metadata about the decoding process
    langfuse_context.update_current_observation(
        output=final_text,
        metadata={
            "draft_length": len(draft_text),
            "final_length": len(final_text),
            "draft_accepted": draft_text == final_text
        }
    )
    
    logger.info("Speculative decoding completed")
    return final_text