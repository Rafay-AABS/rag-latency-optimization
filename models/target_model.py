import os
from groq import Groq
from src.strings import (
    TARGET_MODEL_NAME,
    ENV_GROQ_API_KEY,
    ERROR_GROQ_API_KEY,
    TARGET_MAX_TOKENS,
    TARGET_TEMPERATURE,
    USER_ROLE
)
from src.exceptions import ModelError
from src.logger import logger
from langfuse.decorators import observe, langfuse_context


class TargetModel:
    def __init__(self, model_name: str = TARGET_MODEL_NAME):
        api_key = os.getenv(ENV_GROQ_API_KEY)
        if not api_key:
            raise ModelError(ERROR_GROQ_API_KEY)
        
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        logger.info(f"Initialized target model: {model_name}")

    @observe(as_type="generation")
    def verify(self, prompt: str, draft_text: str) -> str:
        """
        Verify draft output by generating with target model and comparing.
        In API-based speculative decoding, we generate and check if draft matches.
        """
        try:
            logger.debug(f"Verifying with {self.model_name}")
            
            # Update Langfuse context with generation details
            langfuse_context.update_current_observation(
                name="target-model-verification",
                input=prompt,
                model=self.model_name,
                metadata={
                    "max_tokens": TARGET_MAX_TOKENS,
                    "temperature": TARGET_TEMPERATURE,
                    "draft_text_length": len(draft_text)
                }
            )
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": USER_ROLE, "content": prompt}],
                max_tokens=TARGET_MAX_TOKENS,
                temperature=TARGET_TEMPERATURE
            )
            verified_text = response.choices[0].message.content
            
            # Simple verification: if draft is prefix of target output, accept it
            draft_accepted = verified_text.startswith(draft_text.strip())
            if draft_accepted:
                logger.info("Draft accepted by target model")
                final_output = draft_text
            else:
                logger.info("Target model provided different output")
                final_output = verified_text
            
            # Update with generation output and usage
            langfuse_context.update_current_observation(
                output=final_output,
                usage={
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                },
                metadata={
                    "draft_accepted": draft_accepted,
                    "verified_text_length": len(verified_text)
                }
            )
            
            return final_output
        except Exception as e:
            logger.error(f"Target model verification failed: {e}")
            raise ModelError(f"Target model verification failed: {e}") from e
