from fastapi import FastAPI
from pydantic import BaseModel
from app.core.pipeline import SpeculativeRAG

app = FastAPI(title="Speculative RAG API")
rag = SpeculativeRAG()

class Query(BaseModel):
    question: str

@app.post("/ingest")
def ingest(pdf_path: str):
    rag.ingest_pdf(pdf_path)
    return {"status": "ingested"}

@app.post("/ask")
def ask(q: Query):
    return rag.ask(q.question)
