# Deployment

## Jev API

Example `tmux` launch:

```bash
tmux new-session -d -s jev "cd /path/to/Jev && source .venv/bin/activate && python3 -m api.server --port 8030"
```

Stop:

```bash
tmux kill-session -t jev
```

Inspect logs:

```bash
tmux capture-pane -t jev -p
```

## Optional Laya Service

The Laya service is optional and intended for a GPU host. Jev treats it as advisory only: direct command execution remains gated by deterministic router rules.

Install the GPU host dependencies separately, then run:

```bash
uvicorn deploy.gpu2_laya_service:app --host 0.0.0.0 --port 8031
```

Set the API host to use it:

```bash
export JEV_LAYA_URL=http://gpu-host:8031
```

## Production Checklist

- Set explicit `JEV_*` environment variables.
- Keep `JEV_LOG_RAW_QUERY=false` unless raw logs are approved.
- Put the API behind authentication and TLS.
- Monitor `/metrics` and degraded request counts.
- Rebuild vector indexes whenever the embedding model changes.
