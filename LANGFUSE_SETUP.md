# Langfuse Integration Guide

This guide explains how to set up and use Langfuse for observability with your Speculative RAG pipeline.

## What is Langfuse?

Langfuse is an open-source observability and analytics platform for LLM applications. It helps you:
- Track all LLM calls and their performance
- Monitor token usage and costs
- Debug issues with detailed traces
- Analyze query patterns and model behavior

## Setup Langfuse Locally with Docker

### 1. Start Langfuse with Docker

Run Langfuse locally using Docker Compose:

```bash
# Create a directory for Langfuse
mkdir langfuse-local
cd langfuse-local

# Create docker-compose.yml
curl -o docker-compose.yml https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml

# Start Langfuse
docker-compose up -d
```

Alternatively, use the simple Docker command:

```bash
docker run -d --name langfuse \
  -p 3000:3000 \
  -e DATABASE_URL=postgresql://postgres:postgres@db:5432/postgres \
  -e NEXTAUTH_SECRET=mysecret \
  -e SALT=mysalt \
  -e NEXTAUTH_URL=http://localhost:3000 \
  langfuse/langfuse:latest
```

### 2. Access Langfuse UI

Open your browser and navigate to: http://localhost:3000

Create an account or sign in (first time will create a new local account).

### 3. Get API Keys

1. In the Langfuse UI, go to **Settings** → **API Keys**
2. Click **Create new API Key**
3. Copy the:
   - **Public Key** (starts with `pk-lf-`)
   - **Secret Key** (starts with `sk-lf-`)

### 4. Configure Your Application

Create or update your `.env` file:

```bash
# Copy the example file
cp .env.example .env
```

Edit `.env` and add your Langfuse configuration:

```env
# Langfuse Configuration
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key-here
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key-here
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## Using the Integration

### Automatic Tracing

Once configured, all your RAG pipeline operations are automatically traced:

1. **Query Endpoint** - Tracks entire request/response
2. **Pipeline Execution** - Monitors the complete RAG pipeline
3. **Speculative Decoding** - Tracks draft and target model coordination
4. **Draft Model** - Records all draft model generations with token usage
5. **Target Model** - Records verification steps with token usage

### What Gets Tracked

- **Input/Output**: Query text, retrieved documents, generated answers
- **Metadata**: Model names, parameters, chunk counts, query lengths
- **Performance**: Latency for each step in the pipeline
- **Token Usage**: Input/output tokens for draft and target models
- **Success/Failure**: Error tracking and debugging information
- **Speculative Decoding Metrics**: Draft acceptance rate, verification results

### View Traces in Langfuse

1. Run your application: `python app.py` or `uvicorn app:app`
2. Make some queries to your API
3. Open Langfuse at http://localhost:3000
4. Navigate to **Traces** to see all requests
5. Click on any trace to see detailed breakdown

## Example API Usage

```bash
# Upload documents
curl -X POST "http://localhost:8000/upload-documents" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"

# Query (this will be traced in Langfuse)
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the main topics in the documents?"}'
```

## Viewing Langfuse Dashboard

In the Langfuse UI you can see:

### Traces View
- All pipeline executions
- Execution time for each component
- Nested spans showing the flow: Query → Pipeline → Retrieval → Speculative Decode → Draft Model → Target Model

### Generations View
- All LLM generations (draft and target models)
- Token usage and costs
- Model parameters used
- Input/output texts

### Analytics
- Query patterns over time
- Token usage trends
- Model performance comparison (draft vs target)
- Draft acceptance rates

## Disabling Langfuse

If you want to disable Langfuse tracing:

1. Remove the Langfuse environment variables from `.env`, or
2. Simply don't set them - the application will work normally without tracing

The application automatically detects if Langfuse is configured and enables/disables tracing accordingly.

## Docker Compose for Both Services

Here's a complete `docker-compose.yml` to run both Langfuse and your RAG app:

```yaml
version: '3.8'

services:
  langfuse-db:
    image: postgres:15
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: postgres
    volumes:
      - langfuse-data:/var/lib/postgresql/data
    networks:
      - rag-network

  langfuse:
    image: langfuse/langfuse:latest
    ports:
      - "3000:3000"
    environment:
      DATABASE_URL: postgresql://postgres:postgres@langfuse-db:5432/postgres
      NEXTAUTH_SECRET: mysecret
      SALT: mysalt
      NEXTAUTH_URL: http://localhost:3000
    depends_on:
      - langfuse-db
    networks:
      - rag-network

volumes:
  langfuse-data:

networks:
  rag-network:
```

## Troubleshooting

### Traces Not Appearing

1. Check that environment variables are set correctly
2. Verify Langfuse is running: `docker ps`
3. Check application logs for Langfuse initialization messages
4. Ensure network connectivity to http://localhost:3000

### Connection Errors

If you see connection errors:
- Verify Langfuse container is running
- Check Docker network settings
- Try restarting Langfuse: `docker-compose restart`

### Performance Impact

Langfuse tracing adds minimal overhead (<10ms per trace). If you need maximum performance:
- Use async flushing (already configured)
- Disable tracing in production by not setting env vars
- Use sampling (configure in langfuse_integration.py)

## Advanced Configuration

### Custom Sampling

To sample only a percentage of traces, modify [src/langfuse_integration.py](src/langfuse_integration.py):

```python
def initialize_langfuse(host: str, public_key: str, secret_key: str, sample_rate: float = 1.0):
    _langfuse_client = Langfuse(
        host=host,
        public_key=public_key,
        secret_key=secret_key,
        sample_rate=sample_rate  # 0.0 to 1.0
    )
```

### Add Custom Metadata

You can add custom metadata to any trace by updating the code:

```python
langfuse_context.update_current_trace(
    metadata={
        "user_id": user_id,
        "session_id": session_id,
        "custom_field": value
    }
)
```

## Resources

- [Langfuse Documentation](https://langfuse.com/docs)
- [Langfuse GitHub](https://github.com/langfuse/langfuse)
- [Python SDK Documentation](https://langfuse.com/docs/sdk/python)
