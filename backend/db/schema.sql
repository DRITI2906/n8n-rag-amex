-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- for hybrid BM25 / trigram full-text search

-- ─────────────────────────────────────────────────────────
-- Documents: one row per ingested source file
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    doc_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source          TEXT NOT NULL,           -- 'google_drive' | 'slack' | 'email' | 'notion' | 'github' | 'upload'
    file_id         TEXT NOT NULL,           -- external identifier (Drive file ID, URL, etc.)
    file_hash       TEXT NOT NULL UNIQUE,    -- SHA-256 of raw content — deduplication key
    title           TEXT,
    author          TEXT,
    language        TEXT DEFAULT 'en',
    mime_type       TEXT,
    file_url        TEXT,
    creation_date   TIMESTAMPTZ,
    modified_date   TIMESTAMPTZ,
    fetch_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status          TEXT NOT NULL DEFAULT 'pending',  -- 'pending' | 'parsed' | 'chunked' | 'embedded' | 'failed'
    raw_content_size BIGINT,
    metadata        JSONB DEFAULT '{}'::jsonb,        -- extra fields (heading hierarchy, etc.)
    error_message   TEXT
);

CREATE INDEX IF NOT EXISTS idx_documents_source      ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_status      ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_file_hash   ON documents(file_hash);
CREATE INDEX IF NOT EXISTS idx_documents_fetch_ts    ON documents(fetch_timestamp DESC);

-- ─────────────────────────────────────────────────────────
-- Document chunks: text segments after parsing + chunking
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id          UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    chunk_index     INT  NOT NULL,
    chunk_text      TEXT NOT NULL,
    token_count     INT,
    char_start      INT,
    char_end        INT,
    parent_heading  TEXT,
    parent_section  TEXT,
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Full-text search vector (auto-updated via trigger below)
    fts_vector      TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_id     ON document_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_chunks_fts        ON document_chunks USING GIN(fts_vector);
CREATE INDEX IF NOT EXISTS idx_chunks_trgm       ON document_chunks USING GIN(chunk_text gin_trgm_ops);

-- ─────────────────────────────────────────────────────────
-- Chunk embeddings: one row per chunk (vector store)
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    embedding_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id         UUID NOT NULL UNIQUE REFERENCES document_chunks(chunk_id) ON DELETE CASCADE,
    text_hash        TEXT NOT NULL,              -- MD5 of chunk_text — caching key
    embedding_vector vector(384),               -- 384 dims for all-MiniLM-L6-v2; change for OpenAI (1536)
    model_name       TEXT NOT NULL,
    embedding_ts     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- HNSW index for fast ANN search (pgvector >= 0.5)
CREATE INDEX IF NOT EXISTS idx_embeddings_hnsw ON chunk_embeddings
    USING hnsw (embedding_vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS idx_embeddings_text_hash ON chunk_embeddings(text_hash);

-- ─────────────────────────────────────────────────────────
-- Query logs: every /query call is persisted for analytics
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS query_logs (
    query_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          TEXT,
    question         TEXT NOT NULL,
    answer           TEXT,
    sources          JSONB DEFAULT '[]'::jsonb,   -- [{chunk_id, doc_id, score}, ...]
    retrieval_ms     INT,
    generation_ms    INT,
    model_used       TEXT,
    top_k            INT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_query_logs_user    ON query_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_query_logs_created ON query_logs(created_at DESC);

-- ─────────────────────────────────────────────────────────
-- User feedback: thumbs up / down per answer
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id     UUID REFERENCES query_logs(query_id) ON DELETE SET NULL,
    user_id      TEXT,
    rating       SMALLINT NOT NULL CHECK (rating IN (1, -1)),  -- 1=up, -1=down
    comment      TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_query  ON feedback(query_id);
CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback(rating, created_at DESC);

-- ─────────────────────────────────────────────────────────
-- Ingestion log: per-run fetch metadata from n8n
-- ─────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source          TEXT,
    docs_fetched    INT DEFAULT 0,
    docs_added      INT DEFAULT 0,
    chunks_created  INT DEFAULT 0,
    errors          INT DEFAULT 0,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at     TIMESTAMPTZ,
    status          TEXT DEFAULT 'running'   -- 'running' | 'success' | 'partial' | 'failed'
);
