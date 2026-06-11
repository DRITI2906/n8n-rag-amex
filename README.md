# Enterprise Document Intelligence System — n8n + RAG

A production-ready document Q&A system that automatically ingests, indexes, and retrieves information from company documents. Employees ask natural language questions and receive contextually accurate answers with source citations.

**Tech Stack:** FastAPI · PostgreSQL + pgvector · n8n · React 18 · Ollama / Anthropic / OpenAI · Docker

---

## Architecture Overview

```
Documents (Drive, Slack, Notion, Upload)
        │
        ▼
   [n8n Workflow 1 — Daily Sync]
        │
        ├─► POST /parse_document   → text extraction + metadata
        ├─► POST /chunk_document   → recursive chunking (512 tok, 100 overlap)
        └─► POST /embed_chunks     → HuggingFace / OpenAI embeddings → pgvector

   [n8n Workflow 2 — On-Demand Query]
        │
        ▼
   POST /query
        ├─► embed question
        ├─► hybrid search (vector HNSW + full-text, RRF fusion)
        ├─► construct prompt with retrieved chunks
        └─► stream LLM response (SSE)

   [n8n Workflow 3 — Feedback Loop]
        └─► weekly quality report → Slack
```

---

## Services

| Service | URL | Description |
|---|---|---|
| Frontend (React) | http://localhost:3001 | Chat UI |
| Backend (FastAPI) | http://localhost:8000 | RAG API |
| n8n | http://localhost:5678 | Workflow orchestrator |
| Ollama | http://localhost:11434 | Local LLM (Mistral) |
| PostgreSQL | localhost:5432 | Vector + metadata store |

---

## Quick Start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (with Compose v2)
- At least 8 GB RAM free (Ollama + model cache)
- Git

### 1. Clone and configure

```bash
git clone <repo-url>
cd n8n-rag-amex
cp .env.example .env
```

Edit `.env` — the defaults work for local development. Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` if you want cloud LLMs instead of Ollama.

### 2. Start all services

```bash
docker compose up -d
```

First run pulls ~4 GB of images. The backend waits for Postgres to be healthy before starting.

### 3. Pull the LLM model into Ollama

```bash
docker exec rag_ollama ollama pull mistral
```

This downloads ~4 GB once and persists in the `ollama_data` volume.

### 4. Import n8n workflows

1. Open http://localhost:5678 — log in with the credentials in your `.env` (`N8N_USER` / `N8N_PASSWORD`)
2. Go to **Workflows → Import from File**
3. Import each file from `n8n-workflows/`:
   - `workflow1-daily-sync.json`
   - `workflow2-on-demand-query.json`
   - `workflow3-feedback-loop.json`
4. In each workflow, update the **Credentials** nodes with your actual Google Drive / Slack / Notion tokens

### 5. Open the UI

Navigate to http://localhost:3001 — type a question or upload a document to get started.

---

## Environment Variables

```bash
# Database
POSTGRES_USER=raguser
POSTGRES_PASSWORD=ragpassword
POSTGRES_DB=ragdb

# n8n auth
N8N_USER=admin
N8N_PASSWORD=admin123

# Embedding model: "huggingface" (free/local) or "openai"
EMBEDDING_MODEL=huggingface
HF_MODEL_NAME=all-MiniLM-L6-v2

# LLM provider: "ollama" | "anthropic" | "openai"
LLM_PROVIDER=ollama
OLLAMA_MODEL=mistral

# API keys (leave blank if not using)
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-haiku-4-5-20251001
OPENAI_API_KEY=
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-3.5-turbo

# Chunking
CHUNK_SIZE_TOKENS=512
CHUNK_OVERLAP_TOKENS=100

# Retrieval
DEFAULT_TOP_K=5
SIMILARITY_THRESHOLD=0.35

# Slack webhook for n8n notifications (optional)
SLACK_WEBHOOK_URL=
```

---

## API Reference

Base URL: `http://localhost:8000`  
Interactive docs: http://localhost:8000/docs

### `POST /parse_document`

Accepts a file upload, extracts text and metadata, deduplicates by SHA-256 hash.

**Request:** `multipart/form-data` — `file`, `source` (google_drive | slack | email | notion | github | upload), optional `file_id`, `file_url`

**Response:**
```json
{
  "doc_id": "uuid",
  "title": "HR Policies 2024",
  "author": "Jane Smith",
  "language": "en",
  "file_hash": "sha256...",
  "raw_content_size": 42000,
  "already_exists": false,
  "text_preview": "first 200 chars...",
  "metadata": {}
}
```

Supported formats: PDF, DOCX, PPTX, HTML, Markdown, plain text.

---

### `POST /chunk_document`

Splits a parsed document into overlapping chunks using recursive heading → paragraph → sentence splitting.

**Request:**
```json
{ "doc_id": "uuid", "max_tokens": 512, "overlap_tokens": 100 }
```

**Response:**
```json
{ "doc_id": "uuid", "chunks_created": 34, "chunk_ids": ["uuid", ...] }
```

---

### `POST /embed_chunks`

Generates and stores embeddings for chunks. Skips chunks whose text hasn't changed (MD5 cache).

**Request:**
```json
{ "chunk_ids": ["uuid", ...], "batch_size": 64 }
```

Or embed all unembedded chunks in a document: `{ "doc_id": "uuid" }`

**Response:**
```json
{ "embedded": 34, "skipped_cached": 2, "failed": 0 }
```

---

### `POST /search`

Hybrid retrieval: combines pgvector HNSW cosine similarity with PostgreSQL full-text search, fused via Reciprocal Rank Fusion.

**Request:**
```json
{
  "question": "What is the remote work policy?",
  "top_k": 5,
  "similarity_threshold": 0.35,
  "source": "google_drive",
  "date_from": "2024-01-01",
  "use_hybrid": true
}
```

**Response:**
```json
{
  "query": "What is the remote work policy?",
  "results": [
    {
      "chunk_id": "uuid",
      "chunk_text": "...",
      "relevance_score": 0.92,
      "doc_id": "uuid",
      "doc_title": "HR Policies 2024",
      "source": "google_drive",
      "parent_heading": "Work Arrangements"
    }
  ],
  "total": 5
}
```

---

### `POST /query`

Full RAG chain: retrieves chunks then generates an LLM answer. Supports streaming (SSE) and non-streaming modes.

**Request:**
```json
{
  "question": "What is the remote work policy?",
  "top_k": 5,
  "stream": true,
  "use_local_llm": true,
  "provider": "ollama",
  "user_id": "user_123"
}
```

**Non-streaming response:**
```json
{
  "query_id": "uuid",
  "question": "...",
  "answer": "Employees may work remotely up to 3 days per week...",
  "sources": [
    { "doc_id": "...", "doc_title": "HR Policies 2024", "chunk_id": "...", "relevance_score": 0.92 }
  ],
  "retrieval_time_ms": 48,
  "generation_time_ms": 1820,
  "model_used": "mistral"
}
```

**Streaming:** `Accept: text/event-stream` — server sends `metadata`, `token`, and `done` events.

If the LLM fails, the API falls back to returning the ranked chunks with a note.

---

### `POST /query/feedback`

Submit a thumbs-up or thumbs-down rating for an answer.

**Request:**
```json
{ "query_id": "uuid", "rating": 1, "user_id": "user_123", "comment": "Great answer" }
```

`rating`: `1` = thumbs up, `-1` = thumbs down.

---

## n8n Workflows

### Workflow 1 — Daily Document Sync & Index

**Trigger:** Cron — daily at 2:00 AM UTC

Fetches new and modified documents from Google Drive, Slack (pinned messages), and Notion. For each document calls the parse → chunk → embed pipeline sequentially. Logs results to the `ingestion_runs` table and sends a Slack notification summary. Failed documents trigger a separate Slack error alert.

**Setup required:** Configure credentials in n8n for Google Drive OAuth2, Slack API, and Notion API. Set the `BACKEND_URL` variable in the workflow to `http://backend:8000`.

---

### Workflow 2 — On-Demand Query

**Trigger:** Webhook — `POST http://localhost:5678/webhook/ask`

Accepts `{ "question": "...", "user_id": "...", "top_k": 5, "source": null, "provider": "ollama" }`, calls `/query` on the backend, logs the result to PostgreSQL, and returns the answer to the caller. Intended for Slack bot or API integrations.

**Example:**
```bash
curl -X POST http://localhost:5678/webhook/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our parental leave policy?", "user_id": "emp_42"}'
```

---

### Workflow 3 — Quality Feedback Loop

**Trigger A:** Webhook — `POST http://localhost:5678/webhook/feedback`

Receives real-time feedback votes (`{ "query_id": "...", "rating": 1 }`) and stores them via the `/query/feedback` endpoint.

**Trigger B:** Cron — every Monday at 9:00 AM UTC

Queries the last 7 days of feedback, calculates satisfaction rate, identifies the top 5 lowest-rated queries, and posts a summary report to Slack.

---

## Database Schema

All tables live in the `ragdb` PostgreSQL database. The schema is applied automatically on first container start from `backend/db/schema.sql`.

| Table | Purpose |
|---|---|
| `documents` | One row per ingested file. Tracks source, hash, status, metadata. |
| `document_chunks` | Text segments after chunking. Includes FTS vector (auto-generated). |
| `chunk_embeddings` | 384-dim vectors (HNSW indexed) for similarity search. |
| `query_logs` | Every `/query` call with latency and sources. |
| `feedback` | Thumbs up/down ratings linked to query logs. |
| `ingestion_runs` | Per-run stats from n8n Workflow 1. |

**Vector index:** HNSW with `m=16, ef_construction=64` (cosine distance) — targets <200ms for top-10 search.

**Hybrid search:** `chunk_embeddings` (pgvector) + `document_chunks.fts_vector` (GIN-indexed tsvector) + trigram index for BM25-style keyword matching. Results fused with Reciprocal Rank Fusion (k=60).

---

## LLM & Embedding Options

### Embedding models

| Option | Model | Dims | Cost | Quality |
|---|---|---|---|---|
| **Default** | `all-MiniLM-L6-v2` (HuggingFace) | 384 | Free, local | Good |
| Alternative | `text-embedding-3-small` (OpenAI) | 1536 | Paid per token | Better |

To switch: set `EMBEDDING_MODEL=openai` and `OPENAI_API_KEY` in `.env`, then re-embed with `POST /embed_chunks`.

**Note:** Changing the model requires re-embedding all chunks because vector dimensions change. The `chunk_embeddings` table must be truncated and `schema.sql` re-applied with the new `vector(1536)` dimension.

### LLM providers

| Option | Provider | Model | Notes |
|---|---|---|---|
| **Default** | Ollama (local) | Mistral 7B | Private, no cost, ~8 GB RAM |
| Cloud | Anthropic | claude-haiku-4-5-20251001 | Fast, low cost |
| Cloud | OpenAI | gpt-3.5-turbo | Widely used |

Switch per-request by setting `"provider": "anthropic"` in the `/query` body, or globally via `LLM_PROVIDER` in `.env`.

---

## Project Structure

```
n8n-rag-amex/
├── backend/
│   ├── main.py              # FastAPI app + CORS + lifespan
│   ├── config.py            # Pydantic settings
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── db/
│   │   ├── connection.py    # AsyncPG connection pool
│   │   └── schema.sql       # Tables, indexes, pgvector setup
│   ├── routers/
│   │   ├── parse.py         # POST /parse_document
│   │   ├── chunk.py         # POST /chunk_document
│   │   ├── embed.py         # POST /embed_chunks
│   │   ├── search.py        # POST /search
│   │   └── query.py         # POST /query + POST /query/feedback
│   ├── services/
│   │   ├── parser.py        # pdfplumber, python-docx, python-pptx, BS4
│   │   ├── chunker.py       # Recursive heading→paragraph→sentence
│   │   ├── embedder.py      # HuggingFace / OpenAI, MD5 cache
│   │   ├── retriever.py     # Hybrid vector+FTS, RRF fusion
│   │   └── llm.py           # Ollama / Anthropic / OpenAI, SSE streaming
│   └── tests/
│       └── test_chunker.py
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api.js           # queryStream, querySync, submitFeedback, uploadDocument
│   │   └── components/
│   │       ├── ChatInterface.jsx
│   │       ├── FeedbackButtons.jsx
│   │       ├── HistorySidebar.jsx
│   │       └── SourceCard.jsx
│   ├── nginx.conf
│   └── Dockerfile
├── n8n-workflows/
│   ├── workflow1-daily-sync.json
│   ├── workflow2-on-demand-query.json
│   └── workflow3-feedback-loop.json
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## Performance Metrics

The system tracks performance across three categories: latency, retrieval quality, and user satisfaction. All metrics are persisted in PostgreSQL and queryable at any time.

### Latency

Every `/query` call measures and stores two timings using `time.monotonic()`:

| Metric | What it measures | Stored in | Target |
|---|---|---|---|
| `retrieval_ms` | Time to embed the question + run hybrid search + RRF fusion | `query_logs.retrieval_ms` | < 200 ms |
| `generation_ms` | Time from first LLM token request to final token | `query_logs.generation_ms` | < 2800 ms |
| End-to-end | `retrieval_ms + generation_ms` | Computed | < 3 000 ms |

Both values are returned in every API response so the frontend can display them, and are logged to `query_logs` on every call for historical analysis.

**Query latency over time:**
```sql
SELECT
  DATE_TRUNC('hour', created_at) AS hour,
  ROUND(AVG(retrieval_ms))       AS avg_retrieval_ms,
  ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY retrieval_ms)) AS p95_retrieval_ms,
  ROUND(AVG(generation_ms))      AS avg_generation_ms,
  COUNT(*)                       AS queries
FROM query_logs
GROUP BY 1
ORDER BY 1 DESC
LIMIT 24;
```

---

### Retrieval Quality

The retrieval pipeline produces two intermediate scores before returning results:

**Cosine similarity score** (vector search)
- Computed by pgvector as `1 - (embedding_vector <=> query_vector)` — range [0, 1]
- Only chunks scoring ≥ `SIMILARITY_THRESHOLD` (default **0.35**) are considered
- The system over-fetches `top_k × 4` candidates before re-ranking to improve recall

**RRF score** (after hybrid fusion)
- Formula: `Σ 1 / (60 + rank)` across both the vector ranking and the full-text (BM25) ranking
- A chunk appearing in both rankings gets a higher combined score
- The final `relevance_score` in every API response is this RRF value

**Full-text search score** (BM25 via PostgreSQL)
- Computed by `ts_rank(fts_vector, plainto_tsquery('english', question))`
- Catches keyword-heavy queries that pure semantic search can miss (e.g. product names, codes)

**Target:** retrieval recall@5 ≥ 70% (percentage of queries where the correct chunk appears in the top 5 results)

---

### User Satisfaction (Feedback Loop)

Every answer has thumbs-up / thumbs-down buttons. Ratings are stored in the `feedback` table and analysed weekly by n8n Workflow 3.

| Metric | Definition | Computed by |
|---|---|---|
| **Satisfaction rate** | `thumbs_up / total_ratings × 100` | Workflow 3 weekly report |
| **Top 5 failing queries** | Queries with the most thumbs-down votes in the last 7 days | Workflow 3 weekly report |
| **Total feedback volume** | Absolute counts of up/down per week | Workflow 3 weekly report |

The weekly report posts automatically to Slack every Monday at 9 AM UTC.

**Manual query:**
```sql
SELECT
  COUNT(*)                                                            AS total,
  SUM(CASE WHEN rating = 1  THEN 1 ELSE 0 END)                      AS thumbs_up,
  SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END)                      AS thumbs_down,
  ROUND(100.0 * SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) / COUNT(*), 1) AS satisfaction_pct
FROM feedback
WHERE created_at >= NOW() - INTERVAL '7 days';
```

---

### Ingestion Metrics

Every run of Workflow 1 (Daily Sync) writes a row to `ingestion_runs`:

| Column | Meaning |
|---|---|
| `docs_fetched` | Total documents retrieved from all sources |
| `docs_added` | New documents after deduplication (SHA-256 hash check) |
| `chunks_created` | Chunks produced and embedded in this run |
| `errors` | Documents that failed parsing or embedding |
| `status` | `success` / `partial` / `failed` |

**Check recent sync runs:**
```sql
SELECT source, docs_fetched, docs_added, chunks_created, errors, status, started_at
FROM ingestion_runs
ORDER BY started_at DESC
LIMIT 10;
```

---

### Embedding Cache Efficiency

`POST /embed_chunks` reports how many chunks were skipped because their text hash (MD5) already exists in `chunk_embeddings`:

```json
{ "embedded": 30, "skipped_cached": 4, "failed": 0 }
```

A high `skipped_cached` count after re-running the pipeline is expected and correct — it means re-ingesting the same documents does not burn API calls or GPU time.

---

## Running Tests

```bash
docker exec rag_backend python -m pytest tests/ -v
```

The chunker tests cover edge cases: very long paragraphs, nested heading structures, tables, and empty documents.

---

## Common Operations

**Trigger a manual document sync (skip the 2 AM cron):**
In n8n, open Workflow 1 and click **Execute Workflow**.

**Re-embed all chunks after switching embedding model:**
```bash
# Truncate embeddings, then trigger re-embedding
docker exec rag_postgres psql -U raguser -d ragdb -c "TRUNCATE chunk_embeddings;"
curl -X POST http://localhost:8000/embed_chunks \
  -H "Content-Type: application/json" \
  -d '{"doc_id": null}'   # null = all unembedded chunks
```

**Check ingestion logs:**
```bash
docker exec rag_postgres psql -U raguser -d ragdb \
  -c "SELECT source, docs_added, chunks_created, errors, status, started_at FROM ingestion_runs ORDER BY started_at DESC LIMIT 10;"
```

**View top failing queries (last 7 days):**
```bash
docker exec rag_postgres psql -U raguser -d ragdb -c "
SELECT q.question, COUNT(*) AS downvotes
FROM feedback f
JOIN query_logs q ON f.query_id = q.query_id
WHERE f.rating = -1 AND f.created_at > NOW() - INTERVAL '7 days'
GROUP BY q.question
ORDER BY downvotes DESC
LIMIT 5;"
```

**Stop all services:**
```bash
docker compose down
```

**Stop and delete all data (volumes):**
```bash
docker compose down -v
```

---

## Stopping and Cleanup

```bash
# Stop containers, keep data
docker compose down

# Stop and remove all volumes (wipes the database and model cache)
docker compose down -v
```
