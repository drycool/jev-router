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
- `core/decision_engine.py` - optional llama.cpp-style decision endpoint client.
- `core/shadow.py` - fire-and-forget shadow probes for candidate engines (never touch routing).
- `agents/base.py` - general, code, DB, and troubleshooting agents.
- `api/server.py` - FastAPI service and observability endpoints.
- `scripts/build_vector_index.py` - builds the optional `.npz` vector index from LightRAG chunks.
- `scripts/benchmark_laya.py` - scores the Laya tier against a labeled fixture.
- `deploy/gpu2_laya_service.py` - the GPU-side Laya inference service (`:8031`).
- `deploy/laya-ctl.sh` - start/stop/status/warmup/lease helper for that service.
- `deploy/fetch-model.sh` - resumable checkpoint download for a slow HF link.
- `deploy/idle-shutdown.sh` - the GPU2 idle watchdog, lease-aware.

## Features

- Sub-millisecond direct routing for obvious commands.
- SQLite FTS5 local retrieval with rebuild-safe indexing.
- Optional vector search with model and dimension checks.
- LightRAG timeout handling with degraded-mode fallback to local context.
- JSONL decision logging for future evaluation or router distillation.
- Prometheus-compatible `/metrics` endpoint.
- Diagnostic `/decision-test` endpoint for evaluating a llama.cpp-style decision layer without changing production routing.
- **Shadow mode** (`core/shadow.py`): probe a candidate engine on live traffic without letting it steer routing, so it can be judged on real requests instead of a demo.
- Regression tests for routing, fallback, vector-index validation, shadow probes, and API diagnostics.

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

Decision-engine diagnostic probe:

```bash
curl -X POST http://127.0.0.1:8030/decision-test \
  -H 'Content-Type: application/json' \
  -d '{"query":"Which retrieval path should handle this?","candidates":["exact_fts","vector_fast","graph_lightrag"],"schema":"routing_v1"}'
```

This endpoint is intentionally diagnostic: it records metrics and fallback behavior, but it does not alter `/query` routing.

## Shadow Mode

A candidate engine can be probed on live traffic without being allowed to steer anything:

```bash
export JEV_SHADOW_MODE=laya          # or: decision
export JEV_SHADOW_URL=http://192.168.11.87:8031   # defaults per target
export JEV_SHADOW_MAX_INFLIGHT=1
python3 -m api.server --port 8030
```

Per request, `core/shadow.py` schedules one background probe and returns immediately. The probe:

- never raises into the request path and never changes the response;
- is capped at `JEV_SHADOW_MAX_INFLIGHT` concurrent calls — extra work is *skipped and counted*, not queued, so bursts cannot pile up;
- treats an engine's own fallback answer (`status` other than `success`) as an `unavailable` observation rather than a usable one, so a sleeping GPU cannot flatter the numbers;
- writes `{"event": "shadow", ...}` into `jev_decisions.jsonl` and feeds `/stats` + `/metrics`.

```bash
curl -s http://127.0.0.1:8030/stats | python3 -m json.tool | sed -n '/shadow/,$p'
curl -s http://127.0.0.1:8030/metrics | grep jev_shadow
```

## The Laya System-1 Tier (GPU2)

`deploy/gpu2_laya_service.py` exposes Laya over HTTP on `:8031`; the gateway calls it
from `core/router.py`. It never imports PyTorch itself — a slow or dead GPU must not
endanger the Pi's routing loop — and the tier is **advisory only**: it holds no authority
over retrieval, over command execution, or over the subject domain. The classifier is
*started* concurrently with the local index lookup and awaited only if the local path
cannot answer on its own.

```bash
# on GPU2
./deploy/laya-ctl.sh start      # takes a watchdog lease first
./deploy/laya-ctl.sh wait       # blocks until /health says the model is loaded
./deploy/laya-ctl.sh warmup     # pay the first-call CUDA cost before a benchmark
./deploy/laya-ctl.sh status
./deploy/laya-ctl.sh hold       # extend the lease so the idle watchdog spares the box
```

`laya-ctl.sh predict "<query>"` runs a single prediction; `POST /schema` returns the exact
question schema being sent.

### Measured on this hardware (RTX 3060 12 GB, 2026-09-22)

Two checkpoints are preloaded through `laya.Router`, which picks one per request from the
detected script. `scripts/benchmark_laya.py` produced this on the 21-query fixture:

```text
model            answ    strat   domain  conf med   ≥thr   p50 ms   p95 ms
--------------------------------------------------------------------------
english         21/21    66.7%    61.9%       0.1   0.0%     57.9     62.3
multilingual    21/21    33.3%    33.3%       0.5  23.8%     28.6     32.1
regex local         -        -   100.0%         -      -        -        -
```

Read this honestly:

- **Latency** — the multilingual checkpoint answers in ~29 ms (p95 32 ms), comfortably inside
  the budget. The English checkpoint costs ~2x (~58/62 ms) and would *always* exceed a 50 ms
  client timeout, which is why `JEV_LAYA_TIMEOUT_S` is 0.15 rather than 0.05.
- **The gateway's own round trip is ~28.6 ms p50 / 29.4 ms max** measured from the Pi
  (`LayaTier1Client`). Two fixes were needed to get there, both found by measuring rather than
  reasoning:
  - `httpx.Timeout` is *not* a total deadline — it applies per socket read, so a 50 ms setting
    was observed returning after 181 ms. The budget is now enforced with `asyncio.timeout`.
  - the client opened a new TCP connection per call, which cost ~15 ms of the round trip; it now
    pools one connection. Before that, e2e p50 was 44 ms instead of 29 ms.
- **The local regex beats the model at subject domain** (100% vs 33%). Both models answer
  `raspberry_pi` for almost everything, including a car-engine question. Subject domain is not
  what this checkpoint is trained to decide — keep `detect_domain()` for it.
- **`strat`/`domain` accuracy above is measured against labels that are themselves debatable**
  (several retrieval-strategy labels are arguable). Treat it as a smoke test, not a verdict.
- Adding the preset block cost ~12 ms of inference (27.6 → 39 ms average across mixed
  checkpoints); it is still one forward pass, just more head tokens.

### Keeping it off the blocking path

Measured with `scripts/measure_route_latency.py` against a live GPU2:

| query shape | `route()` p50 before | after |
| --- | --- | --- |
| answerable from the local FTS index | 55.98 ms | **0.16 ms** |
| no local hit | 55.62 ms | 56.40 ms |

In the first row, 55.34 ms of the 55.98 ms was the GPU2 round trip, and the answer that came
back never used the verdict — the local index answers itself in ~0.6 ms. On the fall-through
path the classifier now also overlaps the embedding call instead of serialising in front of it
(checked with a 120 ms embedding stub: 123.9 ms total, not 184 ms).

Two things had to be fixed after the first attempt, both found by measuring:

- **Abandoned calls are not free.** Left running fire-and-forget, the requests nobody would read
  queued ahead of the requests that *did* need a verdict on GPU2's single-stream service: the
  fall-through path went from 55.6 ms to 120.4 ms p50, with half its calls timing out. The task
  is now cancelled as soon as the local path answers. Enrichment for such queries is what shadow
  mode is for.
- **Raising the confidence threshold would not have fixed the confident-wrong answers.** The
  service reported `confidence` as the minimum over our two hand-written questions — the
  out-of-distribution taxonomy — and the wrong answer was the confident one (0.9995). The
  threshold now gates the shipped preset `task` signal only; the invented labels are recorded
  for analysis and no longer consulted.
- **One authority was deliberately left in place: `target_agent`.** The `DB` vs `GENERAL` choice
  still follows the classifier's invented `strategy` label, gated on the trusted signal.
  Replacing it changes production semantics and removing it would drop a capability, so it is
  documented here rather than changed quietly. It is the last thing that label acts on.

### Why our own taxonomy had to be replaced

The checkpoint is trained with RL against specific workflows. Asking it our invented
`strategy`/`domain` labels on Russian traffic produced confidences of 0.003–0.37 with
near-random answers — and one confidently wrong answer (`raspberry_pi` at probability 0.9995
for "rewrite the router in Go"). The same model, asked the *shipped* preset
(`laya.router_questions()`), answered `code` at 0.926 and `chitchat` at 0.988 on the same
queries. Evidence and method: `docs/laya-preset-vs-custom-20260922.txt`.

The service therefore sends both blocks and returns both: the shipped preset (`difficulty`,
`task`, `needs_tools`, `is_sensitive`) plus the Jev-specific labels. **Use the preset signals
to drive the strategy decision; do not invent new taxonomies for this model.**

### Routing a non-English deployment

The repo root is the English checkpoint (ModernBERT-large); `multilingual` is a bundled
subfolder (mmBERT-base, 100+ languages). The English checkpoint does not degrade gracefully
off English — it collapses while staying confident, so gating on confidence cannot catch it.
Both are preloaded (`LAYA_PRELOAD=english,multilingual`) so a language flip costs detection
only, not a model reload.

### Operations on GPU2

- `deploy/idle-shutdown.sh` is the idle watchdog. It now honours a **lease**
  (`~/logs/activity.lease`), which the Laya service refreshes on every `/predict` and
  deployments refresh explicitly via `laya-ctl.sh hold`. It also counts established inbound
  SSH connections (utmp misses non-interactive `ssh host cmd`) and long installs running
  outside any SSH session, and it treats recognised model servers as idle while they merely
  hold VRAM. Without the lease, a `tmux`-based install looks idle and the box powers off
  mid-deployment.
- `deploy/fetch-model.sh` downloads a checkpoint resumably. The HF link here runs at
  ~0.7 MB/s and xet stalls on the last file, so one attempt is allowed to run for 30 minutes;
  a short timeout looks like "downloading 0 bytes" while the network is saturated.


## Configuration

All runtime configuration is environment-based. Start from `.env.example`.

Important variables:

- `JEV_LIGHTRAG_API` - LightRAG HTTP endpoint.
- `JEV_LIGHTRAG_CHUNKS_PATH` - source `kv_store_text_chunks.json` used to rebuild FTS5 on startup.
- `JEV_LLM_HOST` and `JEV_LLM_MODEL` - Ollama-compatible generation endpoint and model.
- `JEV_EMBEDDING_API` and `JEV_EMBEDDING_MODEL` - embedding endpoint/model for vector search.
- `JEV_LAYA_URL` - optional remote Laya classifier endpoint.
- `JEV_LAYA_TIMEOUT_S` - total budget for that classifier, enforced with `asyncio.timeout`. Locally answered queries never await it; this bounds what it can add to a fall-through.
- `JEV_LAYA_CONFIDENCE_THRESHOLD` - confidence the shipped preset `task` signal must reach before a verdict counts as accepted. It does not gate the hand-written `strategy`/`domain` questions.
- `JEV_DECISION_ENGINE_URL` - optional llama.cpp-style `/v1/decision` endpoint.
- `JEV_SHADOW_MODE` - `off` (default), `laya`, or `decision`; what to probe in the background.
- `JEV_SHADOW_URL`, `JEV_SHADOW_TIMEOUT_S`, `JEV_SHADOW_MAX_INFLIGHT`, `JEV_SHADOW_SAMPLE_RATE` - shadow probe controls.
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

- `GET /stats` - JSON counters for requests, tiers, degraded requests, agent errors, Laya decisions, and shadow probes.
- `GET /metrics` - Prometheus text exposition, including `jev_shadow_*` families.
- `POST /decision-test` - safe probe for the optional decision engine.
- `jev_decisions.jsonl` - privacy-preserving decision log using query hashes by default; shadow observations are appended as `{"event": "shadow", ...}`.

Each routing event also records `execution.laya_accepted`, i.e. whether the System-1 verdict
would have cleared `JEV_LAYA_CONFIDENCE_THRESHOLD`, and `/stats` reports `laya_not_awaited` for
requests the local path answered without waiting for the classifier. A high `laya_not_awaited`
beside a low `laya_accepted` is the shape to want: the classifier is not earning its latency on
that traffic.

Every successful verdict also records `checkpoint` and `routing_reason`. Those two fields are what
make the threshold measurable per model rather than pooled across two that need not agree.

## Benchmarking the Laya tier

```bash
python3 scripts/benchmark_laya.py --include-baseline                      # routed
python3 scripts/benchmark_laya.py --models english multilingual --repeat 3 # A/B checkpoints
```

It scores strategy and domain accuracy separately, the confidence distribution, how often the
threshold is cleared, latency p50/p95, which checkpoint answered, and the full confusion table.
Every disagreement is printed — a caller cannot be misled by a tidy summary. Add
`--include-baseline` to score the local regex domain classifier on the same fixture, which is
the comparison that matters.

### Setting the threshold from data

The gate can only be set from real traffic, and only per checkpoint — the English and multilingual
models need not report preset confidence on the same scale, and merging them is how a threshold
ends up calibrated for neither:

```bash
python3 scripts/analyze_laya_verdicts.py --threshold 0.90
```

It reads `jev_decisions.jsonl` and reports, per checkpoint: the `task` confidence distribution
(median / p90 / max), the share clearing the gate, the invented confidence for contrast, and the
status breakdown (`not_awaited` / `success` / `timeout`). A checkpoint whose share is ~0 is not
calibrated — it simply never fires, which is a different problem from a threshold set too low.

## Development

```bash
python3 -m unittest discover -v
python3 -m py_compile core/router.py core/decision_engine.py core/shadow.py core/laya_client.py api/server.py scripts/benchmark_laya.py scripts/measure_route_latency.py scripts/analyze_laya_verdicts.py tests/test_router.py
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
