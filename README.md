# nexus-rag (Module 2)

A from-scratch retrieval-augmented generation service. It ingests your documents,
stores their embeddings in Postgres (pgvector), and answers questions by
retrieving the most relevant chunks and having an LLM answer *from that context*.
Generation is delegated to the resilient-llm-gateway, so all provider
fallback and retry logic lives in one place - nexus-rag never calls a model
vendor directly.

## The idea in one line

Index once (chunk -> embed -> store), then per question: embed the question,
find the top-k most similar chunks by cosine distance, stitch them into a prompt,
and ask the gateway to answer using only that context - returning the answer plus
the sources it came from.

## Layout

```
nexus-rag/
├── src/                         # import root (see "Imports" below)
│   ├── core/
│   │   ├── config.py            # typed settings from env / .env
│   │   ├── db.py                # Neon connection pool + pgvector registration
│   │   ├── embeddings.py        # fastembed bge-small (384-dim); text -> vector
│   │   ├── chunking.py          # split a document into overlapping chunks
│   │   ├── store.py             # insert chunks + cosine top-k search
│   │   ├── gateway.py           # httpx client for the gateway's /v1/chat
│   │   └── rag.py               # orchestration: ingest() and answer_question()
│   ├── api/
│   │   ├── health.py            # GET /health
│   │   ├── ingest.py            # POST /documents
│   │   └── query.py             # POST /query
│   └── main.py                  # FastAPI app + lifespan (warm model + pool)
├── migrations/
│   └── 001_init.sql             # CREATE EXTENSION vector; documents table + HNSW index
├── scripts/
│   └── migrate.py               # apply every migrations/*.sql in order
├── tests/                       # (staged)
├── .env.example                 # copy to .env; documents every setting
├── .gitignore
├── pyproject.toml               # pytest pythonpath + ruff config
├── requirements.txt             # runtime deps
└── requirements-dev.txt         # runtime + pytest / ruff
```

## How it works

Two pipelines, both living in `core/rag.py`:

- **Ingestion** (`POST /documents`, `ingest()`): `chunk_text` slices the document
  into overlapping ~1000-char pieces, `embed_texts` turns each into a 384-dim
  vector, and `insert_chunks` writes one row per chunk to Postgres.
- **Query** (`POST /query`, `answer_question()`): `embed_query` embeds the
  question with the *same* model, `search` returns the top-k nearest chunks by
  cosine distance, the chunks are assembled into a context block, and the gateway
  generates a grounded answer. The retrieved chunks are returned as `sources`.

The same embedding model must embed both chunks and queries, or the two never
land in the same vector space and search returns noise.

## Imports

Absolute, rooted at `src/`: `from core.rag import ingest`,
`from api.query import router`. No `app.` prefix, no relative `..` hops. `src/`
is put on the path three ways depending on context:

- **Run:** `uvicorn main:app --app-dir src` (the `--app-dir` flag adds `src/`).
- **Tests:** `pythonpath = ["src"]` in `pyproject.toml`, so no install needed.
- **Scripts:** `scripts/migrate.py` does `sys.path.insert(0, "src")` up top.

## Dependencies

| Package | Role |
|---|---|
| `fastapi` + `uvicorn[standard]` | the web service |
| `pydantic-settings` | typed config from env / `.env` |
| `fastembed` | local embedding model - text -> vector, no API key |
| `psycopg[binary,pool]` | Postgres driver + connection pool for Neon |
| `pgvector` | `vector` type adapter (Python list <-> Postgres vector) |
| `httpx` | HTTP client used to call the gateway |

`requirements.txt` is the runtime set; `requirements-dev.txt` adds `pytest` and
`ruff`.

## Prerequisites

- A **Neon** Postgres database (free tier). Copy its direct connection string
  (`postgresql://...?sslmode=require`) into `.env` as `DATABASE_URL`.
- The **resilient-llm-gateway** running locally on `:8080` for the generation
  step (its own `.env` with `LLM_CHAIN` + provider keys). Ingestion and retrieval
  do **not** need it - only `POST /query` does.

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate                       # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements-dev.txt

cp .env.example .env                          # then paste your Neon DATABASE_URL

python scripts/migrate.py                     # creates the extension, table, and index
uvicorn main:app --app-dir src --reload --port 8081
```

nexus-rag runs on **:8081** (the gateway is on :8080, so both run side by side).
On first start, fastembed downloads the bge-small ONNX model once (~67 MB) and
caches it.

## The two endpoints

### Ingest a document - `POST /documents`

```bash
curl.exe -sS -X POST http://localhost:8081/documents -H "Content-Type: application/json" -d "{\"source\":\"embeddings\",\"text\":\"An embedding maps text to a vector so that similar meanings sit close together. Cosine similarity measures the angle between two embeddings.\"}"
```

```json
{ "source": "embeddings", "chunks_ingested": 1 }
```

`source` is a name/id for the document; `metadata` (optional) is a free-form JSON
object stored alongside each chunk.

### Ask a question - `POST /query`

```bash
curl.exe -sS -X POST http://localhost:8081/query -H "Content-Type: application/json" -d "{\"question\":\"how is similarity between texts measured?\"}"
```

```json
{
  "question": "how is similarity between texts measured?",
  "answer": "Similarity between texts is measured with cosine similarity, which compares the angle between their embedding vectors.",
  "provider": "gemini",
  "model": "gemini-3.6-flash",
  "sources": [
    { "id": 4, "source": "embeddings", "content": "An embedding maps text ...", "similarity": 0.776 },
    { "id": 3, "source": "neon-notes", "content": "Neon is serverless ...",     "similarity": 0.502 }
  ]
}
```

`answer` is grounded in the retrieved `sources`; `provider`/`model` report which
target in the gateway's chain actually served the request. Optional `top_k`
overrides how many chunks are retrieved.

### `GET /health`

```bash
curl.exe -sS http://localhost:8081/health          # {"status":"ok"}
```

## The schema

`migrations/001_init.sql` creates one row per chunk:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id         BIGSERIAL   PRIMARY KEY,
    source     TEXT        NOT NULL,
    content    TEXT        NOT NULL,
    metadata   JSONB       NOT NULL DEFAULT '{}',
    embedding  VECTOR(384) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX documents_embedding_hnsw
    ON documents USING hnsw (embedding vector_cosine_ops);
```

`VECTOR(384)` is locked to the embedding model's dimension. The HNSW index with
`vector_cosine_ops` is what makes `ORDER BY embedding <=> query` fast.

## Configuration

Every setting has an env var (see `.env.example`):

| Var | Default | Meaning |
|---|---|---|
| `DATABASE_URL` | *(required)* | Neon connection string; no default, so a missing one fails fast on startup |
| `GATEWAY_URL` | `http://localhost:8080` | where the gateway is reachable |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | fastembed model |
| `EMBEDDING_DIM` | `384` | must equal the `VECTOR(N)` column |
| `CHUNK_SIZE` | `1000` | chunk length, in characters |
| `CHUNK_OVERLAP` | `150` | characters shared between adjacent chunks |
| `TOP_K` | `5` | chunks retrieved per query |

## Design decisions worth knowing

- **Generation is delegated, not reimplemented.** RAG builds the prompt and calls
  the gateway's `/v1/chat`; the fallback chain and retries stay in one service.
  RAG owns retrieval, the gateway owns resilient model calls.
- **Local embeddings.** fastembed runs a small ONNX model on CPU - no API key, no
  per-embed network call, and it works offline once the model is cached.
- **Neon (serverless Postgres).** Scales to zero when idle, so there's no
  always-on database bill, and it's a real managed Postgres - nothing throwaway.
- **Vectors go through pgvector's `Vector` type.** The store wraps embeddings in
  `pgvector.Vector` so psycopg sends a real `vector` on both writes and queries.
  Passing a plain list makes psycopg send a `double precision[]`, which the
  `<=>` operator has no cast for.
- **Grounding.** The system prompt instructs the model to answer only from the
  provided context and say it doesn't know otherwise; `sources` are returned so an
  answer is traceable to the chunks that produced it.
- **Sync endpoints.** Embedding and DB calls are blocking, so the routes are plain
  `def` - FastAPI runs them in a threadpool and the event loop stays free.

## Not built yet (next slices, in order)

1. Re-ingestion dedup - replace a `source`'s chunks instead of duplicating them.
2. Similarity threshold - drop low-scoring chunks before building the prompt.
3. File ingestion (PDF / markdown / text) instead of raw-text POSTs.
4. bge query-instruction prefix for a retrieval-quality bump.
5. Delete endpoint + retrieval filtered by `source`.
6. Tests (chunking unit tests; ingest -> retrieve integration).
7. Deployment - ECR + an `ecs-service` deployment (RAG public, gateway internal
   via Service Connect); Neon stays the external database, so no vector-DB module
   is needed.
