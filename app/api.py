from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from app.core.pipeline import SpeculativeRAG
from app.core.config import get_settings
from app.core.logging import setup_logging
import logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

# Initialize RAG Pipeline
# We initialize it once at startup
rag = SpeculativeRAG()

class Query(BaseModel):
    question: str

class IngestRequest(BaseModel):
    pdf_path: str

class Response(BaseModel):
    draft: str
    final: str

@app.get("/health")
def health_check():
    return {"status": "ok", "version": settings.APP_VERSION}

@app.post("/ingest")
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest a PDF file. This is a blocking operation, so we might want to run it in background
    or just accept it takes time. For now, we'll run it in background.
    """
    try:
        # Check if file exists
        import os
        if not os.path.exists(request.pdf_path):
            raise HTTPException(status_code=404, detail=f"File not found: {request.pdf_path}")
            
        background_tasks.add_task(rag.ingest_pdf, request.pdf_path)
        return {"status": "ingestion_started", "message": f"Ingestion started for {request.pdf_path}"}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=Response)
def ask(q: Query):
    try:
        result = rag.ask(q.question)
        if "error" in result:
             raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))
