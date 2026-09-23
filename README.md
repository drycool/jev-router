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

Configuration comes from the environment, and `.env` in the project root is loaded at startup
by `core/env.py`. A variable already present in the process environment wins, so
`JEV_X=... python3 -m api.server` overrides the file for a single run.

Important variables:

- `JEV_LIGHTRAG_API` - LightRAG HTTP endpoint.
- `JEV_LIGHTRAG_ENABLED` - `1` (default) calls the graph tier; `0` parks it and serves the same local retrieval instantly. See **Deferred** below.
- `JEV_LIGHTRAG_CHUNKS_PATH` - source `kv_store_text_chunks.json` used to rebuild FTS5 on startup.
- `JEV_LLM_HOST` and `JEV_LLM_MODEL` - Ollama-compatible generation endpoint and model.
- `JEV_EMBEDDING_API` and `JEV_EMBEDDING_MODEL` - embedding endpoint/model for vector search.
- `JEV_VECTOR_TIMEOUT_S` - total budget for the embedding call, enforced with `asyncio.timeout`. On expiry the vector layer is skipped and the local FTS results are served, which is the same outcome as an unreachable embedder.
- `JEV_LAYA_URL` - optional remote Laya classifier endpoint.
- `JEV_LAYA_TIMEOUT_S` - total budget for that classifier, enforced with `asyncio.timeout`. Locally answered queries never await it; this bounds what it can add to a fall-through.
- `JEV_LAYA_CONFIDENCE_THRESHOLD` - confidence the shipped preset `task` signal must reach before a verdict counts as accepted. It does not gate the hand-written `strategy`/`domain` questions.
- `JEV_DECISION_ENGINE_URL` - optional llama.cpp-style `/v1/decision` endpoint.
- `JEV_SHADOW_MODE` - `off` (default), `laya`, or `decision`; what to probe in the background.
- `JEV_SHADOW_URL`, `JEV_SHADOW_TIMEOUT_S`, `JEV_SHADOW_MAX_INFLIGHT`, `JEV_SHADOW_SAMPLE_RATE` - shadow probe controls.
- `JEV_LOG_RAW_QUERY` - keep `false` unless raw user prompts are intentionally logged.

## Deferred (in development)

### LightRAG graph tier - parked until the GPU budget grows

Measured 2026-09-22 and recorded here so it does not have to be re-derived:

| phase | measured |
| --- | --- |
| context only (`only_need_context`) | 10.9 s |
| full query, card free | 40.2 s |
| full query, Laya resident | 62.0 s |

So `JEV_LIGHTRAG_READ_TIMEOUT_S=5` guarantees `lightrag_timeout` on every fall-through
request: the router gives up about 5 s into an 11 s floor, and the request pays for nothing.
The dominant cost is the size of the retrieved context (`top_k=40` returns ~19.6k prompt
tokens, so most of the generation phase is prefill), and `top_k=10` measured 24.3 s end to
end - better, still far outside the budget. Two levers were quantified: the `top_k` table
and the `num_ctx`-as-a-VRAM-decision measurement. Full breakdown, including the
order-dependent nature of the VRAM spill and the fact that LightRAG caches answers by query
text, is in the `rag-query` skill under `references/performance-diagnosis.md`.

**Decision:** optimisation is frozen until GPU VRAM grows. The tier is not deleted - it is
switched off the request path, so the freeze costs nothing while it waits:

```bash
JEV_LIGHTRAG_ENABLED=0   # skip the call, serve the same local retrieval instantly
```

That returns the same context the timeout path already served, with
`fallback_reason=lightrag_disabled` instead of `lightrag_timeout`, and reports
`rag_configuration.lightrag_required=false`. `GET /health` exposes the current state.

Measured on one fall-through query, same answer either way (500-character preview
identical): fresh query 5.45 s -> 0.43 s; already-cached query 0.80 s -> 0.43 s. Worth
being precise about what the switch does *not* do: the fall-through path still calls the
GPU2 embedding API for vector search (~32 ms when the box is healthy, under a 30 s client
timeout), so this is not a "local-only" path. Parking the graph tier removes the call that
was guaranteed to fail; it does not remove the GPU dependency.

**Resume when** the retrieved-context size and the GPU budget are both addressed - not
before. Raising the timeout alone was explicitly rejected: it converts a fast degraded
answer into a 40-62 s wait.

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

The vector index is loaded once and cached in memory, re-read only when the file changes on
disk (mtime + size), so rebuilding it does not require a restart. Loading it per request cost
~310 ms of blocked event loop on this hardware while the cosine search itself takes ~13 ms —
and because the load is synchronous, it also delayed timer callbacks, which is why the
classifier's 150 ms budget was observed firing at ~420 ms.

## Working memory

`JEV_MEMORY_DIR` (default `/home/dry/memory`) holds the readable, diffable copy of what the
project knows: `global/` for environment rules, `projects/` for per-project architecture,
`decisions/` for ADRs. It is indexed into the *same* FTS5 table as the corpus, so one BM25 index
answers both, and the documents are tagged `entity_type='memory'` with their absolute path in
`source` — identifiable and removable without touching the corpus.

The server re-indexes that directory on every start, right after the LightRAG rebuild:

```
[Jev] Indexed 4562 chunks into FTS5
[Jev] Indexed 31 memory chunks from 5 files in /home/dry/memory (replaced 0, 21463 chars)
```

Order matters: the LightRAG step begins with `clear()`, so memory indexed before it would be
deleted on every start. `tests/test_memory_index.py::ServerWiringTests` asserts the order rather
than trusting it.

Why this exists at all: the FTS5 table is a *derived* index, rebuilt from
`JEV_LIGHTRAG_CHUNKS_PATH` on every start. Indexing the directory only by hand meant a restart
took the index from 4593 rows to 4562 with zero memory rows — measured, not assumed:

```bash
sqlite3 storage/jev_fts5.db "DELETE FROM chunks WHERE entity_type='memory';"   # 4593 -> 4562
systemctl --user restart jev.service
sqlite3 storage/jev_fts5.db "SELECT entity_type, COUNT(*) FROM chunks GROUP BY entity_type;"  # memory|31
```

Chunking is by Markdown level-2 heading, each chunk prefixed `"<document title> :: <section>"`,
capped at 1800 characters (under the 2000 the corpus uses, so one document type is not penalised
by BM25's length normalisation alone). `scripts/index_memory.py` is the same code from the
command line — with a dry run and per-file counts — for editing a document on a machine where
the service is not running. Two chunkers would drift; there is one, in `core/memory_index.py`.

Measured through the live service with `execute=false` (retrieval only, no LLM): 19 ms / 24 ms /
43 ms for the systemd, Go-MCP and Jev-limits questions, with the expected file at rank 1 in all
three. The full path with the agent costs ~12.7 s, of which retrieval is 24 ms.

Known limits: the vector index does not cover the memory directory (0 of 4562 entries in
`storage/jev_vectors.npz`) so `search_vector` cannot find it, and memory shares one BM25 index
with the OCR'd manual, which occasionally outranks a memory chunk with short garbage fragments
matching on stop words.

## What the agent actually reads

Three separate limits decided this, and two of them were invisible. The same class of defect —
a hard-coded slice where a policy belonged — appeared at three layers, and fixing one layer at a
time produced no improvement at all, which is the part worth recording.

| Layer | Was | Now | Governs |
|---|---|---|---|
| Router: how many chunks enter the context | `[:3]` | `assemble_context()`, dedup + character budget | `JEV_MAX_CONTEXT_CHARS` |
| Retriever: how deep the pool is | 10 results | 20 results | `JEV_RETRIEVAL_LIMIT` |
| Agent: how much of that context reaches the model | `[:4000]` general, `[:3000]` code/db/troubleshooter | one value | `JEV_LLM_CONTEXT_CHARS` |

### The case that tied them together

`"затяжка болтов головки блока цилиндров момент"`. The agent asked for the torque sequence, said
its context ended, and was right.

- The decision record's `context_chars: 3128` matched `[:3]` exactly, so the router was sending
  three of its ten results — and two of those ten were byte-identical copies, so part of the room
  went to the same page twice.
- The chunk carrying the **complete** sequence (stages а–г: 25 Н·м, then 60° three times) sat at
  **rank 13** of the query's matches. At a pool of 10 it was not in the running at all.
- With the pool at 20 and assembly delivering all 16 unique chunks, that chunk lands at
  **characters 6824–7812** of the assembled context — and every agent was cutting its prompt at
  4000. The context was complete and the model still never saw the stage it needed.

Only after all three were fixed did the answer contain stages а–г, in 10.4 s. Each fix alone was
inert: the budget could not recover a chunk retrieval never returned, and neither could deliver
one the prompt truncated away.

### Deduplication

`storage/jev_fts5.db` is rebuilt from `JEV_LIGHTRAG_CHUNKS_PATH` at startup, and that chunk store
carries the same document twice: 865 groups of byte-identical chunks, 1730 of 4562 rows (19%).
Dedup is exact text with only the edges stripped, deliberately not whitespace-normalised, so
chunks that differ mid-text are not merged. On the query above it drops 4 of 20.

Cleaning the chunk store itself means re-ingesting a document in LightRAG, which also affects the
graph and its caches — a separate decision, not taken here. Deduplicating at the point of
consumption belongs in this repo: identical text is never twice useful, and it costs a set lookup.

### Budget and pool depth, measured together

The saturation point moves with the pool depth, so these two settings are read as a pair.
`scripts/measure_context_budget.py` reports both. On four real queries, unique chunks delivered
out of the pool:

| budget | query 1 | query 2 | query 3 | query 4 |
|---|---|---|---|---|
| 4000 | 4/7 | 6/16 | 7/16 | 4/16 |
| 6000 | 5/7 | 9/16 | 9/16 | 9/16 |
| 8000 | 7/7 | 10/16 | 11/16 | 12/16 |
| 12000 | 7/7 | 13/16 | 15/16 | 16/16 |
| **16000** | **7/7** | **16/16** | **16/16** | **16/16** |
| 20000 | byte-identical to 16000; so is 28000 | | | |

Note the columns: query 1 has only 7 unique chunks in its pool, the others have 16. The default is
**16000**, the measured saturation point with this pool depth, where the pool rather than the
budget becomes the constraint. **12000 was the tempting wrong answer** — it fits every chunk of the
query that started this whole investigation, and still drops 3, 1 and 0 chunks on the other three,
which is the original defect at a smaller scale: a budget quietly cutting context while the answer
looks fine.

The old `[:3]` delivered 1691–3078 characters across those queries; the current default delivers
the entire unique pool of all four.

The cost of the larger context is smaller than it looks and is not the deciding factor. An A/B at
2360 and 7569 characters in, two alternating runs each, came back 5.8/7.3 s and 7.8/9.6 s — a
spread inside one budget as wide as the gap between them, confounded further by longer context
producing longer answers while decode dominates. A direct LLM measurement agrees that prefill is
cheap: 4000 characters in → 16.7 s / 1547 prompt tokens, 12000 in → 10.6 s / 2373.

### The graph tier is exempt

It composes its own context, so neither the dedup nor the budget is applied — rewriting a
synthesised answer is a different decision with a different owner. Note that its output *is*
subject to `JEV_LLM_CONTEXT_CHARS` when the agent builds a prompt, which is why that value should
be raised alongside `top_k` if the graph tier is ever unparked.

### Observability

Decision records carry `context_chars`, `context_chunks_considered`, `context_chunks_used`,
`context_chunks_duplicate` and `context_budget`. `/health` reports all three limits:
`max_context_chars`, `retrieval_limit`, `llm_context_chars`. `POST /query` returns the assembly
block as `context_stats`, and the MCP server prints it with every answer
(`context: 16/20 chunks (4 duplicate dropped) · 11718 chars of 16000 budget`), because a truncated
context and a complete one look identical from the answer alone.

Every decision record carries what happened: `context_chars`, `context_chunks_considered`,
`context_chunks_used`, `context_chunks_duplicate`, `context_budget`. `/health` reports the budget
in force, and `POST /query` returns the same block as `context_stats`. Without these a duplicate
regression would be invisible until an answer quietly got worse.

The graph tier is exempt: it composes its own context, and rewriting a synthesised answer is a
different decision with a different owner.

## Ground truth: what can be labelled and what cannot

The router cannot judge its own answers, so the logs are split by whose account they are.

**`jev_decisions.jsonl` — the router's account.** One record per call, and every record
carries a `decision_id` that is returned in the `/query` response. A `signals` block reports
what the router can observe about itself, and **none of it is a judgement**:

```json
{"schema": "decision_v2", "decision_id": "2ad018dd...", "timestamp": "...",
 "decision": {"strategy": "exact_fts", "confidence": 0.95, ...},
 "signals": {"answered": true, "answer_chars": 2391, "answer_preview": "The user is asking...",
             "context_chars": 500, "tier": "tier2", "lightrag_required": false,
             "lightrag_mode": "skip", "execute_requested": true},
 "execution": {"latency_ms": 15098.4, "degraded": false, "agent_error": false, ...}}
```

`answered`, `tier`, `latency_ms`, `degraded` are signals. Reading them as accuracy would be
the central mistake this design exists to prevent.

**`jev_feedback.jsonl` — the consumer's account.** A verdict arrives after the fact, so it
cannot be a field in an append-only record: a second line for the same `decision_id` would
raise "which line wins?", which is exactly the ambiguity that made `query_hash` unusable for
labelling. It goes to its own file and the two are joined by `decision_id`.

```bash
curl -s -X POST http://127.0.0.1:8030/feedback -H "Content-Type: application/json" \
  -d '{"decision_id":"2ad018dd...","verdict":"rejected","source":"human",
       "comment":"the manual gives the torque for a cold engine; the answer does not say so"}'
```

| verdict | meaning |
|---|---|
| `accepted` | usable as given |
| `partial` | needed correction or further work |
| `rejected` | wrong |

There is deliberately no `unknown`. An abstention carries no signal and would only inflate
the label count. `source` is `human`, `script` or `agent`; several verdicts per decision are
kept, because an agent's guess followed by a human's correction is the normal sequence and
collapsing them would destroy the fact that the human disagreed. Precedence (human over
agent, later over earlier) is applied at read time.

Agents reach the same endpoint through the `jev_feedback` MCP tool, quoting the
`decision_id` printed with every `jev_query` answer.

### Answer previews

`JEV_LOG_ANSWER_PREVIEW` (default `true`, `JEV_PREVIEW_CHARS=400`) writes a bounded
answer and context preview into each decision. Without it a decision can only be labelled
while the answer is still in hand — i.e. never — so the log cannot be labelled at all. The
default is on because the corpus here is a public car manual; set it to `false` to log
signals without content. The length stays either way: it is a signal about the answer, not
the answer itself.

### Labelling

The person who owns the labels gets a queue, not a JSON body to hand-write:

```bash
python3 scripts/label.py              # interactive: [a]ccepted [r]ejected [p]artial [s]kip [q]uit
python3 scripts/label.py --list 20    # just show what is awaiting a verdict
python3 scripts/label.py --id <decision_id> --verdict rejected --comment "..." --query "..." --source human
```

Skipping is a first-class option: a guessed label is worse than a missing one, because it
looks like data. The queue never offers a decision that already has a verdict, and it never
offers a v1 record — those have no `decision_id`, so a verdict could not attach to one.

`--query` is worth using. The decision log keeps only a **hash** of the query, so it shows what
was answered but not what was asked, and nobody can judge an answer's correctness without the
question. The reviewer has it in hand at the moment of judging, so attaching it there makes the
label self-contained without turning on raw-query logging. The report counts these separately:

```
  labels carrying their question       3 / 3
```

### Is there enough to act on?

```bash
python3 scripts/label_coverage.py                 # coverage, label trust, readiness verdict
python3 scripts/label_coverage.py --json
```

The report never prints accuracy: a percentage computed from a handful of self-reported
labels would look like a metric and mean nothing. It reports what is real — how many
decisions have a verdict, whether any verdict class is below a usable floor, and whether
cheap agent labels have ever been checked against a human. Its first run, honestly:

```
decisions            52
  by schema          {'decision_v1': 51, 'decision_v2': 1}
  no decision_id     51  <- v1 records, unlabellable forever
feedback entries     2
  orphans            1  <- judged an id the log does not have

label coverage
  labelled           1 / 52  (1.9%)

dataset readiness
  Not usable yet. 1 labelled decisions, but these verdict
  classes are below the floor of 20: {'rejected': 1}
```

The `51 unlabellable` line is not a bug: records written before `decision_id` existed can
never be labelled, because a verdict has nothing to attach to. This is the argument for
adding the field now rather than after the log grows.

### Test runs are redirected away from these logs

A test that posts to `/query` appends a real record, because the endpoint cannot tell a
fixture from traffic. It happened: 20 of the 71 records in `jev_decisions.jsonl` (28%) were
the fixture string `"Raspberry Pi cable"` from `ShadowProbeTests`, all with empty keywords
and entities. Any statistic over that log was inflated by them, and a dataset built from it
would have been trained on test scaffolding.

`tests/__init__.py` now redirects `JEV_DECISION_LOG_PATH` and `JEV_FEEDBACK_LOG_PATH` to a
temp directory before any test module is imported, so no test — including ones written later
by someone who never reads that file — can reach the production logs. The redirect is
asserted by `TestProductionLogsAreProtected`, so removing it fails loudly. Existing
contamination is removed with:

```bash
python3 scripts/quarantine_queries.py --query "Raspberry Pi cable"          # dry run
python3 scripts/quarantine_queries.py --query "Raspberry Pi cable" --apply  # moves to .quarantine
```

## Running as a service

`deploy/jev.service` is a systemd **user** unit, installed as `~/.config/systemd/user/jev.service`:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/jev.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now jev.service
systemctl --user status jev.service
journalctl --user -u jev.service -f
```

A user unit needs `loginctl enable-linger $USER` to start without a login; it is already `yes`
on this machine. Without linger, `systemctl --user` units exist only while the user is logged in,
which looks identical to working until the first reboot.

This matters because the router previously ran in a **tmux session**, which is not a startup
mechanism. A reboot took it down and nothing restarted it or reported it gone — the first sign
was a consumer's tool call failing. If a service is meant to be reachable, its startup has to be
owned by init, not by a terminal that happens to be open.

The same reboot also took down the LightRAG API on `:8020`, which this router calls in its graph
tier. It is the same failure and it has its own unit, kept with that service rather than here:
`/home/dry/LightRag/deploy/lightrag.service`, installed as `~/.config/systemd/user/lightrag.service`.

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

Run it exactly like that, from the project root. Adding `-s tests` looks equivalent and is not:
discovery then imports the modules as top-level (`test_router` rather than `tests.test_router`),
so `tests/__init__.py` never runs — and that file is where the log paths are redirected away from
production. The run then appends the `"Raspberry Pi cable"` fixture from `test_router.py` to
`jev_decisions.jsonl` (one ~912-byte record per run) and `TestProductionLogsAreProtected` fails.
That guard is doing its job: it is how the wrong invocation was caught, not a symptom of a bug in
the router. Records already written are removable by hash:

```bash
python3 scripts/quarantine_queries.py --query "Raspberry Pi cable" --apply
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
