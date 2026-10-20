# Running Skynet_DS

Commands only, grouped by what you're trying to do. All commands run from
the repo root (`E:\Skynet_DS`) unless noted.

## 0. One-time setup

```bash
# Python deps - src/requirements.txt is the real one (Dockerfile uses it).
# The root requirements.txt is stale/unused, ignore it.
pip install -r src/requirements.txt

# Env vars
cp .env.example .env
# then fill in GITHUB_TOKEN, GROQ_API_KEY, etc. - see .env.example comments
# for what each one gates.
```

Key vars from `.env.example` / `src/core/config.py`:

| Var | Needed for |
|---|---|
| `GITHUB_TOKEN` | Pulling PR metadata/diffs (`debug`/`compare` with a real repo, the PR-review pipeline, demo scripts) |
| `GROQ_API_KEY` | LLM-backed analysis pass in the specialist agents; unset = regex-only (fails open) |
| `TELEMETRY_BACKEND` | `fake` (default, canned data) or `wandb` (needs `WANDB_API_KEY`) |
| `PROMETHEUS_URL` | Drift checks / monitoring sweep; unset = skipped |
| `GITHUB_WEBHOOK_SECRET` | Verifying inbound GitHub webhook signatures |
| `SLACK_WEBHOOK_URL` | Drift/deploy-regression alerts; unset = log only |

## 1. Core CLI (`src/cli.py`)

The actual product's entry point - no demo dependencies.

```bash
python -m src.cli debug --run-id run-004
python -m src.cli compare --runs run-baseline,run-004 --metric eval_accuracy
python -m src.cli drift --model-id fraud-clf      # needs PROMETHEUS_URL
```

`debug`/`compare` work out of the box against `FakeTelemetryAdapter`
(`TELEMETRY_BACKEND=fake`, the default) - no external services needed.

## 2. Tests

```bash
python -m pytest tests/           # core product
pytest demo/tests/                # demo layer (separate rootdir/conftest)
python -m evals.run_eval          # labeled-incident precision/latency gate
```

## 3. Webhook service (the real product, per README.md)

```bash
uvicorn src.api.webhooks:app --reload --port 8000
```

Sends a `ping` or `pull_request`/`deployment_status` payload to
`POST /webhooks/github` to exercise it manually (e.g. with `curl` or
GitHub's "Redeliver" button on a configured webhook).

### Via Docker Compose (same thing, containerized)

```bash
docker compose up --build
```

Reads `.env` via `env_file`, persists `data/prs` and `data/repos` in the
`skynet-data` volume. Serves on `localhost:8000`.

## 4. Demo layer (`demo/`) - presentation only, not the core product

```bash
pip install -r demo/requirements-demo.txt

# Full pipeline against a real PR: multi-agent review (Langfuse-traced)
# + Evidently drift dashboard, printed to stdout.
python -m demo.run_demo owner/repo 42

# Same, and also post the review as a real comment on the PR (needs a
# write-scope GITHUB_TOKEN - see PR_Review_Setup_Guide.docx).
python -m demo.run_demo owner/repo 42 --post
```

Optional, for the Langfuse trace UI (`demo/langfuse_tracing.py`, used
automatically by `run_demo` if configured):

```bash
# set before running demo.run_demo
export LANGFUSE_PUBLIC_KEY=...
export LANGFUSE_SECRET_KEY=...
export LANGFUSE_HOST=...   # omit for Langfuse Cloud
```

## 5. Demo dashboard UI (`demo-ui/`, NestJS)

```bash
cd demo-ui
npm install
npm run start:dev      # dev, with reload
npm run build && npm run start:prod   # production build
```

## 6. Knowledge graph (`graphify-out/`)

```bash
graphify .                 # full rebuild
graphify . --update        # re-extract only changed files
graphify query "<question>"   # ask about the codebase
```
