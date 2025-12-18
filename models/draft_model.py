import os
from groq import Groq
from src.strings import (
    DRAFT_MODEL_NAME,
    ENV_GROQ_API_KEY,
    ERROR_GROQ_API_KEY,
    DRAFT_MAX_TOKENS,
    DRAFT_TEMPERATURE,
    USER_ROLE
)
from src.exceptions import ModelError
from src.logger import logger
from langfuse.decorators import observe, langfuse_context


class DraftModel:
    def __init__(self, model_name: str = DRAFT_MODEL_NAME):
        api_key = os.getenv(ENV_GROQ_API_KEY)
        if not api_key:
            raise ModelError(ERROR_GROQ_API_KEY)
        
        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        logger.info(f"Initialized draft model: {model_name}")

    @observe(as_type="generation")
    def generate(self, prompt: str, max_new_tokens: int = DRAFT_MAX_TOKENS) -> str:
        """
        Generate draft response using faster, smaller model via API.
        """
        try:
            logger.debug(f"Generating draft with {self.model_name}")
            
            # Update Langfuse context with generation details
            langfuse_context.update_current_observation(
                name="draft-model-generation",
                input=prompt,
                model=self.model_name,
                metadata={
                    "max_tokens": max_new_tokens,
                    "temperature": DRAFT_TEMPERATURE
                }
            )
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": USER_ROLE, "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=DRAFT_TEMPERATURE
            )
            text = response.choices[0].message.content
            logger.debug(f"Draft generated: {len(text)} characters")
            
            # Update with generation output and usage
            langfuse_context.update_current_observation(
                output=text,
                usage={
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0
                }
            )
            
            return text
        except Exception as e:
            logger.error(f"Draft model generation failed: {e}")
            raise ModelError(f"Draft model generation failed: {e}") from e
