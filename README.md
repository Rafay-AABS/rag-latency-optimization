# Speculative RAG Pipeline for Latency Optimization

A high-performance Retrieval-Augmented Generation (RAG) system that leverages speculative decoding techniques to minimize latency while maintaining answer quality. This project uses fast language models for initial draft generation and more capable models for verification, all while providing comprehensive observability through Langfuse.

## Key Features

- **Speculative RAG Architecture**: Combines fast draft generation with quality verification to optimize response times
- **Multi-Model Approach**: Uses Groq's Llama models for both drafting (fast) and verification (accurate)
- **Local Embeddings**: HuggingFace embeddings with ChromaDB for efficient vector storage
- **Observability**: Full tracing and monitoring with Langfuse
- **PDF Processing**: Automated ingestion and chunking of PDF documents
- **Latency Optimization**: Designed to reduce response times compared to traditional single-model RAG systems

## Architecture

```
PDF Document → Text Splitting → Embedding (HuggingFace) → ChromaDB Vector Store
                                      ↓
User Query → Retrieval → Draft Generation (Groq Llama 3.1 8B) → Verification (Groq Llama 3.3 70B) → Final Answer
                                      ↓
                                 Langfuse Tracing
```

## Prerequisites

- Python 3.9+ (recommended: 3.11 or 3.12, avoid 3.14)
- Docker & Docker Compose (for Langfuse)
- API Keys:
  - Groq API Key (for both drafting and verification models)

## Quick Start

1. **Clone and Setup Environment**:
   ```bash
   git clone <repository-url>
   cd rag-latency-optimization
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Unix/Mac:
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Environment Configuration**:
   ```bash
   cp .env.example .env
   ```
   Fill in your `.env` file:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
   LANGFUSE_SECRET_KEY=your_langfuse_secret_key
   LANGFUSE_HOST=http://localhost:3000
   ```

3. **Start Langfuse (Optional for Tracing)**:
   ```bash
   docker-compose up -d
   ```
   Access Langfuse at `http://localhost:3000` and create API keys.

4. **Run the Pipeline**:
   ```bash
   python main.py path/to/your/document.pdf
   ```

## Usage

### Interactive Mode

```bash
python main.py
# Enter PDF path when prompted
```

### Direct Mode

```bash
python main.py /path/to/document.pdf
```

Once loaded, ask questions about your document:

```
User: What are the main findings in this paper?
--- Draft (Groq) ---
[Fast initial response]

--- Final (Groq Verified) ---
[Refined, verified answer]
```

## How It Works

1. **Document Ingestion**: PDFs are loaded, split into chunks, and embedded using local HuggingFace models stored in ChromaDB
2. **Speculative Generation**:
   - **Draft Phase**: Fast Groq Llama 3.1 8B model generates initial answers based on retrieved context
   - **Verification Phase**: More capable Groq Llama 3.3 70B model reviews and refines the draft for accuracy
3. **Tracing**: All interactions are logged to Langfuse for performance monitoring and debugging

## Performance Benefits

- **Reduced Latency**: Speculative approach provides faster initial responses
- **Quality Assurance**: Two-stage verification ensures accuracy
- **Cost Efficiency**: Balances speed and quality across different model capabilities
- **Observability**: Comprehensive tracing helps identify bottlenecks

## Configuration

### Model Selection

The pipeline uses two Groq models:
- **Drafter**: `llama-3.1-8b-instant` (fast, cost-effective)
- **Verifier**: `llama-3.3-70b-versatile` (accurate, thorough)

### Chunking Parameters

- Chunk Size: 1000 characters
- Overlap: 200 characters

### Embedding Model

- Model: `all-MiniLM-L6-v2` (local, no API costs)

## Troubleshooting

### Common Issues

- **Python 3.14 Errors**: Use Python 3.11 or 3.12. Version 3.14 lacks binary wheels for some dependencies.
- **Rate Limits**: Groq has rate limits; consider upgrading your plan for production use.
- **Memory Issues**: Large PDFs may require more RAM; consider chunk size adjustments.

### Environment Setup

If you encounter dependency issues:
```bash
# Recreate virtual environment
rm -rf .venv
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Dependencies

Key libraries:
- `langchain`: RAG pipeline framework
- `langchain-groq`: Groq model integration
- `langchain-huggingface`: Local embeddings
- `chromadb`: Vector database
- `langfuse`: Observability and tracing
- `pypdf`: PDF processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license here]

## Acknowledgments

- Groq for providing fast inference APIs
- Langfuse for observability tools
- LangChain for the RAG framework
