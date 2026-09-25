
# Zepto GenAI Service

A small RAG-based GenAI service for answering questions
about Zepto's policy documents.

## Features

- Document ingestion and chunking
- Local embeddings using all-MiniLM-L6-v2
- ChromaDB vector storage
- LangGraph StateGraph workflow
- Deterministic offline mock LLM mode
- Structured Pydantic output
- FastAPI API
- Docker support

## Project Architecture

### 1. Ingestion

The eight policy documents are stored in the `docs/`
directory.

The `load_and_chunk_documents()` function in
`src/rag.py` reads the documents and creates one
chunk per document.

### 2. Embedding

The `SentenceTransformer` model
`all-MiniLM-L6-v2` in `src/rag.py` converts
each document chunk into an embedding vector.

### 3. Storage

ChromaDB stores the document chunks and their
embeddings in the collection:

`zepto_policy_chunks`

The collection is persisted under:

`data/chroma/`

### 4. Retrieval

The `retrieve_chunks()` function in `src/rag.py`
embeds the user query and retrieves the top 3
most similar chunks using cosine distance.

### 5. LangGraph workflow

The workflow is implemented in `src/graph.py`.

Nodes:

- `classify_intent`
- `retrieve_and_answer`
- `direct_answer`

The `classify_intent` node classifies the query
as `policy_question` or `general_question`.

Policy questions go to `retrieve_and_answer`.
General questions go to `direct_answer`.

### 6. Generation

When `MOCK_LLM` is unset or set to `1`:

- Classification uses the required keyword heuristic.
- Policy questions retrieve the top 3 chunks.
- The answer is generated using a deterministic
  template from the top chunk.
- General questions return a fixed canned answer.
- No external LLM API is called.

When `MOCK_LLM=0`:

- The real-LLM extension is selected.
- Retrieval still uses local embeddings and ChromaDB.
- Real LLM integration is optional and is not required
  for the graded baseline.

## Architecture Diagram

User Query
    |
    v
FastAPI /ask
    |
    v
LangGraph classify_intent
    |
    +---------------------------+
    |                           |
    v                           v
policy_question          general_question
    |                           |
    v                           v
Retrieve top 3            direct_answer
    |
    v
Mock or real generation
    |
    v
Pydantic validation
    |
    v
JSON response

## Installation

Create a virtual environment:

`python -m venv .venv`

Activate it:

Windows:
`.venv\Scripts\activate`

Install dependencies:

`pip install -r requirements.txt`

## Build the index

`python -m src.rag`

## Run locally

`uvicorn src.main:app --reload`

Swagger UI:

`http://127.0.0.1:8000/docs`

## Docker

Build:

`docker build -t zepto-genai .`

Run:

`docker run --rm -p 7860:7860 zepto-genai`

## Example API Calls

### Policy question

Request:

`POST /ask`

```json
{
  "query": "What is the delivery fee?"
}
```

Raw response:

{"answer":"Based on the retrieved context: Delivery Policy: \"Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order vo","sources":["doc_01_chunk_00","doc_05_chunk_00","doc_02_chunk_00"],"confidence":1.0}

### General question

Request:

`POST /ask`

```json
{
  "query": "What is the capital of France?"
}
```

Raw response:

{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}

## Mock Mode

The graded baseline uses mock mode:

`MOCK_LLM=1`

No external LLM provider is required.

## Optional Real LLM Extension

The optional real-LLM integration is enabled with:

`MOCK_LLM=0`

Any API key must be stored securely as an environment
variable and must not be committed to the repository.