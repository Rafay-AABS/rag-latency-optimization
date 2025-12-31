from app.core.pipeline import SpeculativeRAG
from scripts.benchmark import benchmark

rag = SpeculativeRAG()
rag.ingest_pdf("sample.pdf")

queries = [
    "What is the main contribution?",
    "Explain speculative decoding"
]

benchmark(rag, queries)
