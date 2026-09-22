# jev-router

`jev-router` is a lightweight multi-tier routing gateway for RAG and agent workflows. It is designed to answer simple requests locally, use fast text/vector retrieval when possible, and escalate only harder questions to LightRAG and an LLM-backed agent.

## Architecture

```text
User query
  -> Tier 1: deterministic fast router
  -> Tier 2a: SQLite FTS5 search
  -> Tier 2b: optional vector search
  -> Tier 3: LightRAG graph retrieval
  -> Tier 4: agent execution through an Ollama-compatible LLM API
```

Core components:

- `core/router.py` - routing pipeline, FTS5/vector retrieval, LightRAG fallback handling.
- `core/laya_client.py` - optional remote Laya/System-1 classifier client with strict timeout.
- `agents/base.py` - general, code, DB, and troubleshooting agents.
- `api/server.py` - FastAPI service and observability endpoints.
- `scripts/build_vector_index.py` - builds the optional `.npz` vector index from LightRAG chunks.
- `deploy/gpu2_laya_service.py` - optional GPU-side Laya inference service.

## Features

- Sub-millisecond direct routing for obvious commands.
- SQLite FTS5 local retrieval with rebuild-safe indexing.
- Optional vector search with model and dimension checks.
- LightRAG timeout handling with degraded-mode fallback to local context.
- JSONL decision logging for future evaluation or router distillation.
- Prometheus-compatible `/metrics` endpoint.
- Regression tests for routing, fallback, vector-index validation, and API diagnostics.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-dev.txt
cp .env.example .env
python3 -m unittest discover -v
python3 -m api.server --host 0.0.0.0 --port 8030
```

Health check:

```bash
curl http://127.0.0.1:8030/health
```

Route without executing an agent:

```bash
curl 'http://127.0.0.1:8030/route-only?query=Raspberry%20Pi%20cable'
```

Full query:

```bash
curl -X POST http://127.0.0.1:8030/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"Какие кабели обсуждались для Raspberry Pi 5?","execute":true}'
```

## Configuration

All runtime configuration is environment-based. Start from `.env.example`.

Important variables:

- `JEV_LIGHTRAG_API` - LightRAG HTTP endpoint.
- `JEV_LIGHTRAG_CHUNKS_PATH` - source `kv_store_text_chunks.json` used to rebuild FTS5 on startup.
- `JEV_LLM_HOST` and `JEV_LLM_MODEL` - Ollama-compatible generation endpoint and model.
- `JEV_EMBEDDING_API` and `JEV_EMBEDDING_MODEL` - embedding endpoint/model for vector search.
- `JEV_LAYA_URL` - optional remote Laya classifier endpoint.
- `JEV_LOG_RAW_QUERY` - keep `false` unless raw user prompts are intentionally logged.

## Indexes

Generated indexes are intentionally ignored by Git:

- `storage/jev_fts5.db`
- `storage/jev_vectors.npz`

FTS5 is rebuilt on API startup from `JEV_LIGHTRAG_CHUNKS_PATH` when that file exists.

Build the optional vector index:

```bash
python3 scripts/build_vector_index.py \
  --chunks /path/to/kv_store_text_chunks.json \
  --output storage/jev_vectors.npz
```

The router rejects stale vector indexes when the embedding model or vector dimension does not match.

## Observability

- `GET /stats` - JSON counters for requests, tiers, degraded requests, agent errors, and Laya decisions.
- `GET /metrics` - Prometheus text exposition.
- `jev_decisions.jsonl` - privacy-preserving decision log using query hashes by default.

## Development

```bash
python3 -m unittest discover -v
python3 -m py_compile core/router.py api/server.py tests/test_router.py
```

## Publication Notes

Do not publish local `.env`, generated indexes, logs, or decision JSONL files. They may contain local paths, operational metadata, or private context.

The intended GitHub repository is:

```text
https://github.com/drycool/jev-router
```

Publish from this local checkout:

```bash
git remote add origin https://github.com/drycool/jev-router.git
git push -u origin main
```
