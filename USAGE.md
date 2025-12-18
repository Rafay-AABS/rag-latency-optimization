# Speculative RAG - Usage Guide

A production-ready Retrieval-Augmented Generation (RAG) system using speculative decoding for improved performance. This system processes PDF documents, builds vector indices, and answers queries using a fast draft model combined with a powerful target model.

## Features

- 🚀 **Speculative Decoding**: Combines fast draft model with powerful target model for optimal speed/quality balance
- 📄 **PDF Processing**: Automatic parsing and text extraction from PDF documents using PyMuPDF
- 🔍 **Semantic Search**: FAISS-based vector search for relevant document retrieval
- 💾 **Smart Caching**: Automatic caching of embeddings to avoid reprocessing
- 🎯 **Interactive Mode**: Continuous Q&A sessions with your documents
- ⚙️ **Configurable**: Extensive CLI options for customization
- 📊 **Production-Ready**: Comprehensive logging, error handling, and validation
- 🔒 **Type-Safe**: Full type hints across the codebase

## Quick Start

### 1. Setup Environment

Make sure you have activated your virtual environment and installed dependencies:

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Install dependencies (if not already done)
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_token_here  # Optional
```

Get your free Groq API key at: https://console.groq.com

### 3. Add PDF Files

Place your PDF files in the `data/raw/` directory.

### 4. Run the Application

```powershell
# Interactive mode (default)
python main.py

# Single query
python main.py --query "What are the key points in these documents?"
```

## Usage Examples

### Basic Usage

```powershell
# Interactive mode with all PDFs in data directory
python main.py --interactive

# Single query with specific PDFs
python main.py --query "Summarize the main ideas" --pdf document1.pdf document2.pdf

# Use PDFs from custom directory
python main.py --query "What is discussed?" --data-dir path/to/pdfs
```

### Advanced Options

```powershell
# Rebuild index (ignore cache)
python main.py --rebuild --query "Explain the concepts"

# Custom model selection
python main.py --query "Analyze" --draft-model llama-3.1-8b-instant --target-model llama-3.3-70b-versatile

# Adjust RAG parameters
python main.py --query "What is this about?" --chunk-size 500 --chunk-overlap 100 --top-k 10

# Enable debug logging
python main.py --verbose --query "Debug this"

# Custom vector store location
python main.py --vector-store my_index --query "Test"
```

### Interactive Mode Example

```powershell
python main.py --interactive
```

Example session:
```
============================================================
Speculative RAG Configuration
============================================================
Embedding Model:    all-mpnet-base-v2
Draft Model:        llama-3.1-8b-instant
Target Model:       llama-3.3-70b-versatile
Chunk Size:         300 words
Chunk Overlap:      50 words
Top-K Results:      5
Vector Store:       vector_store
Data Directory:     data/raw
Interactive Mode:   True
Rebuild Index:      False
============================================================

============================================================
Interactive Mode - Enter your queries (type 'quit' or 'exit' to stop)
============================================================

Query: What are the main topics covered?

--- Answer ---
[Response here]

Query: Can you elaborate on the first point?

--- Answer ---
[Response here]

Query: quit
Exiting interactive mode...
```

## Command-Line Options

### Input Options

- `-q QUERY, --query QUERY`: Query string to process
- `--pdf FILE [FILE ...]`: Specific PDF files to process
- `--data-dir DIR`: Directory containing PDF files (default: `data/raw`)

### Mode Options

- `-i, --interactive`: Run in interactive mode for multiple queries
- `--rebuild`: Force rebuild of vector index (ignore cache)

### Model Configuration

- `--embedding-model MODEL`: Embedding model name (default: `all-mpnet-base-v2`)
- `--draft-model MODEL`: Draft model name (default: `llama-3.1-8b-instant`)
- `--target-model MODEL`: Target model name (default: `llama-3.3-70b-versatile`)

### RAG Parameters

- `--chunk-size N`: Number of words per chunk (default: 300)
- `--chunk-overlap N`: Number of overlapping words (default: 50)
- `--top-k N`: Number of documents to retrieve (default: 5)

### Output Options

- `--vector-store DIR`: Directory for vector store (default: `vector_store`)
- `--log-level LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `--verbose`: Enable verbose output (equivalent to --log-level DEBUG)

## Configuration

### Centralized Strings

All string constants, error messages, and default values are centralized in `src/strings.py`:

```python
# File paths
DATA_RAW_DIR = "data/raw"
VECTOR_STORE_DIR = "vector_store"

# Model names
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
DRAFT_MODEL_NAME = "llama-3.1-8b-instant"
TARGET_MODEL_NAME = "llama-3.3-70b-versatile"

# Parameters
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K_RESULTS = 5
```

Modify these values to change default behavior.

## Caching

The system automatically caches embeddings to improve performance:

- Cached based on document content and configuration
- Stored in `vector_store/cache_metadata.json`
- Use `--rebuild` to force regeneration
- Clear cache by deleting the vector store directory

Cache hits are logged:
```
INFO - Cache hit for key abc123_def456
```

## Logging

Logs are stored in `logs/app_YYYYMMDD.log`:

- **INFO**: General progress and status
- **DEBUG**: Detailed execution information (use `--verbose`)
- **WARNING**: Non-critical issues (e.g., failed PDF parsing)
- **ERROR**: Errors that didn't stop execution
- **CRITICAL**: Fatal errors

View logs in real-time:
```powershell
Get-Content logs\app_20251121.log -Wait
```

## Error Handling

The application provides clear error messages:

- **ValidationError**: Invalid input (empty query, missing files, etc.)
- **ConfigurationError**: Configuration issues (missing API key, invalid parameters)
- **PDFParsingError**: PDF reading/parsing failures
- **EmbeddingError**: Embedding generation failures
- **RetrievalError**: Document retrieval failures
- **ModelError**: Model inference failures

All errors are logged with full context.

## Performance Tips

1. **Use cache**: Don't use `--rebuild` unless necessary
2. **Adjust chunk size**: Larger chunks = fewer embeddings but less granular
3. **Tune top-k**: More documents = better context but slower
4. **Choose models wisely**: Balance speed vs. quality
5. **Process PDFs once**: Cache will handle subsequent queries

## Troubleshooting

### Issue: "No PDF files found"
**Solution**: 
- Ensure PDFs are in `data/raw/` directory
- Check file extensions are `.pdf`
- Try specifying files: `--pdf document.pdf`

### Issue: "GROQ_API_KEY not found"
**Solution**:
- Create `.env` file with: `GROQ_API_KEY=your_key`
- Verify key at https://console.groq.com

### Issue: "Failed to load FAISS index"
**Solution**:
- Delete `vector_store/` directory
- Run with `--rebuild` flag

### Issue: Slow performance
**Solution**:
- Check logs for "Cache hit" message
- Reduce `--top-k` value
- Use smaller chunk size
- Ensure cache is enabled (don't use `--rebuild`)

### Issue: PDF parsing fails
**Solution**:
- Check if PDF is valid and not corrupted
- Ensure PDF contains extractable text (not just images)
- Check logs for specific error details

## Development Dependencies

Install development tools:

```powershell
pip install -r requirements-dev.txt
```

Includes:
- pytest (testing)
- black (code formatting)
- flake8, pylint (linting)
- mypy (type checking)
- ipython, jupyter (interactive development)

## Project Structure

```
speculative-RAG/
├── main.py                    # Main entry point
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Development dependencies
├── .env                       # Environment variables (create this)
├── data/
│   └── raw/                  # Place your PDF files here
├── vector_store/             # FAISS indices and cached embeddings
├── logs/                     # Application logs
├── src/
│   ├── cache.py             # Cache management system
│   ├── chunker.py           # Text chunking utilities
│   ├── cli.py               # Command-line interface
│   ├── config.py            # Configuration management
│   ├── embedder.py          # Embedding generation
│   ├── exceptions.py        # Custom exception classes
│   ├── logger.py            # Logging configuration
│   ├── pdf_parser.py        # PDF parsing with PyMuPDF
│   ├── pipeline.py          # Main RAG pipeline
│   ├── rag_prompt.py        # Prompt template building
│   ├── retriever.py         # Document retrieval
│   ├── speculative_decoder.py  # Speculative decoding logic
│   ├── strings.py           # Centralized string constants
│   └── validators.py        # Input validation utilities
└── models/
    ├── draft_model.py       # Fast draft model (Groq API)
    └── target_model.py      # Accurate target model (Groq API)
```

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Enable verbose mode: `--verbose`
3. Review error messages carefully
4. Check API key and network connectivity
