# Speculative RAG Pipeline with Groq, Gemini, and Langfuse

This project implements a "Speculative RAG" pipeline that uses:
- **Groq (Llama 3)** for fast draft generation.
- **Google Gemini** for embeddings and answer verification/refinement.
- **Langfuse** for observability and tracing (running locally).

## Prerequisites

- Python 3.9+
- Docker & Docker Compose (for Langfuse)
- API Keys:
  - Groq API Key
  - Google Gemini API Key

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Environment Variables**:
    - Copy `.env.example` to `.env`.
    - Fill in your API keys.
    ```bash
    cp .env.example .env
    ```
    - `GROQ_API_KEY`: Your Groq API key.
    - `GOOGLE_API_KEY`: Your Google Gemini API key.
    - `LANGFUSE_...`: Your Langfuse credentials (see below).

3.  **Start Langfuse (Locally)**:
    - Run the docker compose file:
    ```bash
    docker-compose up -d
    ```
    - Open `http://localhost:3000` in your browser.
    - Create an account/project.
    - Create new API keys (Public Key, Secret Key).
    - Update your `.env` file with these keys and `LANGFUSE_HOST=http://localhost:3000`.

## Usage

Run the main script:

```bash
python main.py [path_to_pdf]
```

Or simply:
```bash
python main.py
```
And enter the path when prompted.

## How it Works

1.  **Ingestion**: The PDF is loaded, split, and embedded using **Gemini Embeddings** into a local **ChromaDB**.
2.  **Drafting**: When you ask a question, **Groq (Llama 3)** generates a quick draft answer based on retrieved context.
3.  **Verification**: **Gemini** acts as a verifier, checking the draft against the context to ensure accuracy and refining it if necessary.
4.  **Tracing**: All steps are logged to **Langfuse** for inspection.
