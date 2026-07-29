-- pgvector: adds the VECTOR type and similarity operators. Neon supports it.
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per CHUNK: its text, where it came from, and its embedding.
CREATE TABLE IF NOT EXISTS documents (
    id         BIGSERIAL   PRIMARY KEY,
    source     TEXT        NOT NULL,               -- which file/doc this chunk came from
    content    TEXT        NOT NULL,               -- the chunk text itself
    metadata   JSONB       NOT NULL DEFAULT '{}',  -- anything extra (page, title, ...)
    embedding  VECTOR(384) NOT NULL,               -- MUST equal EMBEDDING_DIM
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- HNSW index for fast approximate nearest-neighbour search.
-- vector_cosine_ops is the cosine variant — it pairs with the <=> operator
-- we'll use at query time. Build is instant on an empty table.
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw
    ON documents USING hnsw (embedding vector_cosine_ops);