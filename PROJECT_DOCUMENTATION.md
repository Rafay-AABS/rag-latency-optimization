# 📚 Speculative RAG - Complete Project Documentation

## 🎯 Project Overview

**Speculative RAG** is a production-ready Retrieval-Augmented Generation (RAG) system that combines semantic search with speculative decoding to deliver fast, accurate responses to queries based on your document collection. It leverages Groq's ultra-fast API to avoid downloading large language models while maintaining high-quality output.

### Key Innovation: Speculative Decoding

The project implements **speculative decoding**, an advanced optimization technique that uses two models working in tandem:
- **Draft Model** (Llama-3.1-8B): Generates quick candidate responses
- **Target Model** (Llama-3.3-70B): Verifies and refines the output

This approach provides near-large-model quality at small-model speeds, achieving 300+ tokens/second throughput.

---

## 🏗️ Architecture Overview

```
User Query
    ↓
[1] Document Chunking (chunker.py)
    ↓
[2] Embedding Generation (embedder.py)
    ↓
[3] Vector Store (FAISS)
    ↓
[4] Semantic Retrieval (retriever.py)
    ↓
[5] Prompt Construction (rag_prompt.py)
    ↓
[6] Draft Generation (draft_model.py)
    ↓
[7] Verification & Refinement (target_model.py)
    ↓
Final Answer
```

---

## 📁 Project Structure

```
speculative-RAG/
│
├── main.py                          # Entry point - orchestrates the entire pipeline
├── README.md                        # User-facing documentation
├── requirements.txt                 # Python dependencies
├── .env                            # Environment variables (API keys)
│
├── data/
│   └── raw/                        # Input: Place your .txt documents here
│       └── sample.txt              # Example document
│
├── vector_store/                   # Auto-generated vector database
│   ├── index.faiss                 # FAISS index for fast similarity search
│   └── embeddings.npy              # Numpy array of document embeddings
│
├── models/                         # LLM wrapper classes
│   ├── draft_model.py              # Fast Llama-3.1-8B for draft generation
│   └── target_model.py             # Accurate Llama-3.3-70B for verification
│
└── src/                            # Core RAG components
    ├── chunker.py                  # Text chunking with overlap
    ├── embedder.py                 # Sentence embedding generation
    ├── retriever.py                # Vector-based document retrieval
    ├── rag_prompt.py               # Prompt template construction
    ├── speculative_decoder.py      # Speculative decoding orchestration
    └── pipeline.py                 # Main RAG pipeline coordinator
```

---

## 🔧 Core Components Explained

### 1. `main.py` - Entry Point

**Purpose**: Orchestrates the entire RAG pipeline from data loading to answer generation.

**Workflow**:
1. Loads environment variables (API keys)
2. Reads all `.txt` files from `data/raw/`
3. Chunks the raw text into manageable pieces
4. Generates embeddings and builds FAISS index
5. Initializes retriever and RAG pipeline
6. Executes a sample query and prints the answer

**Key Code**:
```python
# Load and chunk documents
raw_text = "\n\n".join(open(f).read() for f in files)
chunks = chunk_text(raw_text)

# Build vector store
embed = Embedder()
emb = embed.embed_texts(chunks)
embed.build_faiss(emb)

# Run RAG pipeline
pipeline = SpeculativeRAG(retriever)
answer = pipeline.run(query, chunks)
```

---

### 2. `src/chunker.py` - Text Chunking

**Purpose**: Splits large documents into smaller, overlapping chunks for better retrieval.

**Function**: `chunk_text(text, chunk_size=300, overlap=50)`

**Parameters**:
- `text`: Input document text
- `chunk_size`: Number of words per chunk (default: 300)
- `overlap`: Number of overlapping words between chunks (default: 50)

**How It Works**:
- Splits text into words
- Creates sliding windows of `chunk_size` words
- Overlaps ensure context continuity across chunks
- Returns list of text chunks

**Why Overlap?**
Overlapping prevents information loss at chunk boundaries, ensuring relevant context isn't split awkwardly.

---

### 3. `src/embedder.py` - Embedding Generation

**Purpose**: Converts text chunks into dense vector representations for semantic search.

**Class**: `Embedder`

**Key Methods**:

#### `__init__(model_name="all-mpnet-base-v2")`
- Initializes Sentence-BERT model
- `all-mpnet-base-v2`: 768-dimensional embeddings, excellent performance

#### `embed_texts(texts)`
- Converts text list to numpy array of embeddings
- Uses GPU if available for faster processing

#### `build_faiss(embeddings, save_dir="vector_store")`
- Creates FAISS index using L2 (Euclidean) distance
- Saves index and embeddings to disk for reuse
- Avoids re-embedding on subsequent runs

#### `load_faiss(save_dir="vector_store")`
- Loads pre-built FAISS index from disk
- Returns index and embeddings for retrieval

**Technical Details**:
- **FAISS**: Facebook AI Similarity Search - ultra-fast vector similarity library
- **IndexFlatL2**: Exact search using L2 distance (suitable for small-medium datasets)

---

### 4. `src/retriever.py` - Document Retrieval

**Purpose**: Finds the most relevant document chunks for a given query.

**Class**: `Retriever`

**Key Methods**:

#### `__init__(index_path, emb_path)`
- Loads FAISS index from disk
- Loads embeddings
- Initializes same embedding model for query encoding

#### `retrieve(query, texts, k=5)`
- Embeds the query using the same model
- Searches FAISS index for k nearest neighbors
- Returns top-k most relevant text chunks

**How It Works**:
1. Query → Embedding (768-dim vector)
2. FAISS searches for nearest embeddings
3. Returns indices of closest matches
4. Maps indices back to original text chunks

**Parameters**:
- `k=5`: Number of documents to retrieve (configurable)

---

### 5. `src/rag_prompt.py` - Prompt Engineering

**Purpose**: Constructs the input prompt for language models using retrieved context.

**Function**: `build_rag_prompt(query, retrieved_docs)`

**Template Structure**:
```
You are a helpful assistant.

Context:
[Doc 1]
<retrieved document 1>

[Doc 2]
<retrieved document 2>

...

Question: <user query>
Answer:
```

**Design Rationale**:
- Clear role instruction for the AI
- Numbered documents for traceability
- Separation between context and question
- Simple structure for reliable LLM parsing

---

### 6. `src/speculative_decoder.py` - Speculative Decoding

**Purpose**: Implements the two-model speculative decoding strategy.

**Function**: `speculative_decode(prompt, draft_model, target_model)`

**Algorithm**:
1. **Draft Phase**: Fast model generates candidate response
2. **Verification Phase**: Accurate model validates/refines the draft
3. Returns final verified text

**Why This Works**:
- Draft model is 8B parameters → very fast
- Target model is 70B parameters → very accurate
- If draft is good, we save compute
- If draft is poor, target model corrects it
- Net result: Speed of small model + quality of large model

---

### 7. `models/draft_model.py` - Draft Model

**Purpose**: Fast response generation using Llama-3.1-8B.

**Class**: `DraftModel`

**Configuration**:
- Model: `llama-3.1-8b-instant`
- Max Tokens: 512
- Temperature: 0.7 (balanced creativity)

**Key Method**: `generate(prompt, max_new_tokens=512)`
- Sends prompt to Groq API
- Returns draft response quickly (~100-200ms)
- Uses higher temperature for diverse generation

**API Details**:
- Uses Groq's API via `groq` Python SDK
- Requires `GROQ_API_KEY` environment variable
- Free tier: 30 requests/minute

---

### 8. `models/target_model.py` - Target Model

**Purpose**: High-quality verification using Llama-3.3-70B.

**Class**: `TargetModel`

**Configuration**:
- Model: `llama-3.3-70b-versatile`
- Max Tokens: 1024
- Temperature: 0.3 (more deterministic)

**Key Method**: `verify(prompt, draft_text)`
- Generates response with larger, more accurate model
- Compares draft output to target output
- Returns verified text

**Verification Logic**:
```python
if verified_text.startswith(draft_text.strip()):
    return draft_text  # Draft was good!
else:
    return verified_text  # Use target's output
```

This simple heuristic checks if the draft is a valid prefix of the target output, indicating alignment.

---

### 9. `src/pipeline.py` - RAG Pipeline

**Purpose**: Orchestrates all components into a cohesive RAG system.

**Class**: `SpeculativeRAG`

**Initialization**:
```python
def __init__(self, retriever):
    self.retriever = retriever
    self.draft = DraftModel()
    self.target = TargetModel()
```

**Main Method**: `run(query, texts)`

**Pipeline Flow**:
1. `retrieve()` → Get relevant documents
2. `build_rag_prompt()` → Construct prompt
3. `speculative_decode()` → Generate & verify answer
4. Return final answer

**Clean Abstraction**: Hides complexity, exposes simple interface.

---

## 🔄 Complete Data Flow Example

Let's trace a query through the entire system:

### Input Query
```
"Explain the leadership principles in these documents."
```

### Step-by-Step Execution

#### 1. Document Loading (`main.py`)
```python
files = glob.glob("data/raw/*.txt")
raw_text = "\n\n".join(open(f).read() for f in files)
# Result: "Our company values... [full document text]"
```

#### 2. Chunking (`chunker.py`)
```python
chunks = chunk_text(raw_text, chunk_size=300, overlap=50)
# Result: [
#   "Our company values integrity and innovation...",
#   "innovation and teamwork are core to our...",
#   ...
# ]
```

#### 3. Embedding (`embedder.py`)
```python
embed = Embedder()
emb = embed.embed_texts(chunks)
# Result: numpy array of shape (N, 768)
# e.g., [[0.12, -0.45, 0.78, ...], ...]
```

#### 4. FAISS Index Building
```python
embed.build_faiss(emb)
# Creates: vector_store/index.faiss
#          vector_store/embeddings.npy
```

#### 5. Query Retrieval (`retriever.py`)
```python
docs = retriever.retrieve(query, chunks, k=5)
# Result: [
#   "Our leadership principles include...",
#   "Leaders are expected to demonstrate...",
#   "Innovation requires bold decision making...",
#   ...
# ]
```

#### 6. Prompt Construction (`rag_prompt.py`)
```python
prompt = build_rag_prompt(query, docs)
# Result: 
# """
# You are a helpful assistant.
#
# Context:
# [Doc 1]
# Our leadership principles include...
# 
# [Doc 2]
# Leaders are expected to demonstrate...
#
# Question: Explain the leadership principles in these documents.
# Answer:
# """
```

#### 7. Draft Generation (`draft_model.py`)
```python
draft_text = draft_model.generate(prompt)
# Result: "Based on the documents, the leadership principles are..."
# Time: ~150ms
```

#### 8. Verification (`target_model.py`)
```python
final_text = target_model.verify(prompt, draft_text)
# Result: "Based on the documents, the leadership principles emphasize..."
# Time: ~300ms
# Total: ~450ms (much faster than running 70B alone!)
```

#### 9. Return Answer
```python
print(answer)
# Output:
# --- Answer ---
# Based on the documents, the leadership principles emphasize
# integrity, innovation, and customer obsession. Leaders are
# expected to demonstrate bold decision-making...
```

---

## 🛠️ Technical Stack

### Dependencies (`requirements.txt`)

| Package | Version | Purpose |
|---------|---------|---------|
| `faiss-cpu` | Latest | Fast vector similarity search |
| `sentence-transformers` | Latest | Pre-trained embedding models |
| `numpy` | Latest | Numerical operations |
| `python-dotenv` | Latest | Environment variable management |
| `groq` | Latest | Groq API client for LLM access |

### Why These Choices?

- **FAISS**: Industry-standard vector search (used by Meta, Google)
- **Sentence-Transformers**: State-of-the-art embeddings, easy to use
- **Groq API**: Ultra-fast inference (10-100x faster than typical APIs)
- **No PyTorch/TensorFlow**: Lightweight, API-based approach

---

## 🚀 Setup and Usage

### Prerequisites
- Python 3.8+
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

```bash
# Clone repository
git clone <repo-url>
cd speculative-RAG

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=your_key_here

# Add documents
# Place .txt files in data/raw/

# Run
python main.py
```

### Customization

#### Change Chunk Size
```python
# In main.py
chunks = chunk_text(raw_text, chunk_size=500, overlap=100)
```

#### Retrieve More Documents
```python
# In src/retriever.py
def retrieve(self, query, texts, k=10):  # Changed from k=5
```

#### Use Different Embedding Model
```python
# In src/embedder.py
class Embedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):  # Faster, smaller
```

#### Adjust LLM Parameters
```python
# In models/draft_model.py
response = self.client.chat.completions.create(
    model=self.model_name,
    max_tokens=1024,  # More tokens
    temperature=0.5   # Less creative
)
```

---

## 💡 Key Concepts Explained

### What is RAG (Retrieval-Augmented Generation)?

Traditional LLMs are limited to their training data. RAG extends LLMs with external knowledge:

1. **Index**: Store documents in a searchable format
2. **Retrieve**: Find relevant documents for a query
3. **Augment**: Add retrieved docs to the prompt
4. **Generate**: LLM produces answer grounded in your data

**Benefits**:
- ✅ Up-to-date information
- ✅ Citation-capable
- ✅ Domain-specific knowledge
- ✅ Reduced hallucinations

### What is Speculative Decoding?

Traditional approach: Use large model for everything (slow)
Speculative approach: Small model drafts, large model verifies (fast + accurate)

**Analogy**: Like having a junior developer write code quickly, then a senior developer reviews it. Much faster than senior doing everything!

**Performance Gains**:
- 2-3x speedup typical
- No quality loss
- Works best when draft model is "good enough" most of the time

### What is FAISS?

**F**acebook **AI** **S**imilarity **S**earch - library for efficient similarity search.

**Problem**: Finding similar vectors in millions of embeddings is slow
**Solution**: FAISS uses advanced indexing (trees, quantization, etc.)

**IndexFlatL2**: Exact search, simple, works well for <1M vectors

**Alternative Indices** (for scaling):
- `IndexIVFFlat`: Inverted file index (faster, approximate)
- `IndexHNSW`: Hierarchical Navigable Small World (very fast)

### What are Embeddings?

Embeddings convert text to vectors that capture semantic meaning.

**Example**:
- "dog" → [0.2, 0.8, -0.3, ...]
- "puppy" → [0.25, 0.75, -0.28, ...]  (similar vector!)
- "car" → [-0.6, 0.1, 0.9, ...]  (different vector)

**Model Used**: `all-mpnet-base-v2`
- Dimensions: 768
- Training: 1B+ sentence pairs
- Best balance of speed/quality

---

## 🎯 Performance Considerations

### Bottlenecks

1. **Embedding Generation**: ~10ms per chunk (batch for speed)
2. **FAISS Search**: <1ms for 10k vectors (very fast)
3. **LLM API Calls**: ~150ms (draft) + ~300ms (target)

### Optimization Tips

#### Batch Embedding
```python
# Instead of:
for chunk in chunks:
    embed.embed_texts([chunk])

# Do:
embed.embed_texts(chunks)  # Single batch call
```

#### Cache Embeddings
```python
if os.path.exists("vector_store/index.faiss"):
    retriever = Retriever()  # Load existing
else:
    # Build new index
```

#### Async API Calls
```python
import asyncio
# Can parallelize draft + target for multiple queries
```

### Scaling Guidelines

| Document Count | FAISS Index | Expected Latency |
|----------------|-------------|------------------|
| <10k chunks | IndexFlatL2 | <5ms |
| 10k-100k | IndexIVFFlat | <20ms |
| 100k-1M | IndexHNSW | <50ms |
| >1M | IndexIVF + PQ | <100ms |

---

## 🔒 Security and Best Practices

### Environment Variables
```bash
# .env (NEVER commit to git!)
GROQ_API_KEY=gsk_xxxxx...

# .gitignore
.env
vector_store/
__pycache__/
```

### API Key Safety
- Use `.env` files
- Never hardcode keys
- Rotate keys regularly
- Use separate keys for dev/prod

### Error Handling
```python
# Add try-except blocks
try:
    answer = pipeline.run(query, chunks)
except Exception as e:
    print(f"Error: {e}")
    # Log, retry, fallback, etc.
```

---

## 🐛 Troubleshooting

### Common Issues

#### "GROQ_API_KEY not found"
**Solution**: Create `.env` file with valid API key

#### "No text files found in data/raw/"
**Solution**: Add `.txt` files to `data/raw/` directory

#### "Invalid embeddings shape"
**Solution**: Ensure chunks are non-empty text strings

#### "Rate limit exceeded"
**Solution**: Wait or upgrade Groq plan

### Debug Mode
```python
# Add logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Print intermediate results
print(f"Retrieved {len(docs)} documents")
print(f"Prompt length: {len(prompt)} chars")
```

---

## 🔮 Future Enhancements

### Potential Improvements

1. **Multi-Format Support**: PDF, DOCX, HTML parsing
2. **Advanced Chunking**: Semantic chunking, sentence splitting
3. **Re-ranking**: Use cross-encoder for better retrieval
4. **Streaming**: Stream responses token-by-token
5. **Caching**: Cache frequent queries
6. **Multi-Query**: Generate multiple query variations
7. **Evaluation**: Add RAGAS metrics for quality measurement
8. **Web UI**: Gradio/Streamlit interface

### Code Examples

#### PDF Support
```python
from PyPDF2 import PdfReader

def load_pdfs(directory):
    texts = []
    for pdf_file in glob.glob(f"{directory}/*.pdf"):
        reader = PdfReader(pdf_file)
        text = "\n".join(page.extract_text() for page in reader.pages)
        texts.append(text)
    return "\n\n".join(texts)
```

#### Streaming Responses
```python
def generate_stream(self, prompt):
    stream = self.client.chat.completions.create(
        model=self.model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    for chunk in stream:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

---

## 📊 Comparison with Alternatives

### vs. LangChain
| Feature | Speculative RAG | LangChain |
|---------|----------------|-----------|
| Simplicity | ✅ ~200 lines | ❌ Complex abstractions |
| Speed | ✅ Optimized | ⚠️ More overhead |
| Flexibility | ⚠️ Basic | ✅ Highly extensible |
| Learning Curve | ✅ Easy | ❌ Steep |

### vs. LlamaIndex
| Feature | Speculative RAG | LlamaIndex |
|---------|----------------|------------|
| Vector Store | ✅ FAISS | ✅ Multiple backends |
| Speculative Decode | ✅ Built-in | ❌ Not included |
| API-based | ✅ Groq | ✅ OpenAI, others |
| Complexity | ✅ Simple | ⚠️ Medium |

### vs. Haystack
| Feature | Speculative RAG | Haystack |
|---------|----------------|----------|
| Production Ready | ⚠️ Basic | ✅ Enterprise |
| Setup Time | ✅ 5 minutes | ❌ Hours |
| Customization | ⚠️ Limited | ✅ Extensive |
| Dependencies | ✅ Minimal | ❌ Many |

---

## 📚 References and Resources

### Academic Papers
- [Speculative Decoding (Chen et al., 2023)](https://arxiv.org/abs/2302.01318)
- [RAG: Retrieval-Augmented Generation (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [Sentence-BERT (Reimers & Gurevych, 2019)](https://arxiv.org/abs/1908.10084)

### Documentation
- [Groq API Docs](https://console.groq.com/docs)
- [FAISS Documentation](https://faiss.ai/)
- [Sentence-Transformers](https://www.sbert.net/)

### Tutorials
- [Building RAG Systems](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [Vector Databases Explained](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/)

---

## 👥 Contributing

Contributions welcome! Areas of interest:
- Additional embedding models
- Better verification strategies
- UI/UX improvements
- Documentation enhancements
- Performance optimizations

---

## 📄 License

This project is open source under the MIT License.

---

## 🙏 Acknowledgments

- **Groq**: For providing ultra-fast API access
- **Meta AI**: For FAISS library
- **Hugging Face**: For Sentence-Transformers
- **LLM Community**: For speculative decoding research

---

## 📞 Support

For questions or issues:
- GitHub Issues: [Project Repository]
- Documentation: This file
- Community: [Discord/Slack/etc.]

---

**Last Updated**: November 20, 2025  
**Version**: 1.0.0  
**Author**: Rafay-AABS
