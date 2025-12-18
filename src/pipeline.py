from typing import List
from src.rag_prompt import build_rag_prompt
from src.speculative_decoder import speculative_decode
from src.logger import logger
from models.draft_model import DraftModel
from models.target_model import TargetModel
from langfuse.decorators import observe, langfuse_context


class SpeculativeRAG:
    def __init__(self, retriever):
        logger.info("Initializing SpeculativeRAG pipeline")
        self.retriever = retriever
        self.draft = DraftModel()
        self.target = TargetModel()
        logger.info("Pipeline initialized successfully")

    @observe()
    def run(self, query: str, texts: List[str]) -> str:
        """Run the complete RAG pipeline."""
        logger.info(f"Running pipeline for query: {query[:100]}...")
        
        # Add metadata to the trace
        langfuse_context.update_current_trace(
            name="speculative-rag-pipeline",
            user_id="system",
            metadata={"query_length": len(query), "num_texts": len(texts)}
        )
        
        # Retrieve relevant documents
        docs = self.retriever.retrieve(query, texts)
        logger.debug(f"Retrieved {len(docs)} documents")
        
        # Add retrieval metadata
        langfuse_context.update_current_observation(
            metadata={"num_docs_retrieved": len(docs)}
        )
        
        # Build prompt
        prompt = build_rag_prompt(query, docs)
        logger.debug("Built RAG prompt")
        
        # Generate answer using speculative decoding
        answer = speculative_decode(prompt, self.draft, self.target)
        logger.info("Generated answer successfully")
        
        # Add output metadata
        langfuse_context.update_current_trace(
            output=answer,
            metadata={"answer_length": len(answer)}
        )
        
        return answer
