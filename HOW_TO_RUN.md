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

### Via Docker (everything, containerized)

| Image | Dockerfile | Runs |
|---|---|---|
| `skynet-ds/core` | `Dockerfile` | webhook + worker + `/results` API on :8000; also the CLI and evals |
| `skynet-ds/dashboard` | `demo-ui/Dockerfile` | dashboard on :3000, reading `/results` over HTTP |
| `skynet-ds/demo` | `demo/Dockerfile` | `demo.run_demo` (Langfuse-traced review + Evidently HTML) |
| `prom/prometheus` | (upstream) | :9090, scrape config in `prometheus.yml` |

```bash
cp .env.example .env            # optional; set RESULTS_API_TOKEN before exposing anything
docker compose up --build       # webhook :8000, dashboard :3000, prometheus :9090

docker compose run --rm cli debug --run-id run-004               # core CLI
docker compose run --rm --entrypoint python cli -m evals.run_eval # incident eval
docker compose run --rm demo owner/repo 42                       # demo layer; HTML lands in ./demo/out
```

`cli` and `demo` are behind compose profiles (`tools`, `demo`), so `up`
starts only the long-running services. All Python images run as a non-root
user; webhook, dashboard and Prometheus have healthchecks, and the dashboard
waits for the webhook to be healthy. `skynet-data` persists `data/prs`,
`data/monitoring` and `data/repos`; `prometheus-data` persists the TSDB.

### Durable queue (Celery + Redis, optional)

By default jobs run on an in-process asyncio queue: no Redis, but anything
queued (including the 1-hour post-deploy check) dies with the API process.
To make jobs survive restarts:

```bash
# .env
CELERY_BROKER_URL=redis://redis:6379/0

docker compose --profile queue up --build   # adds redis + a celery worker (-B = scheduler)
# or locally, with your own Redis:
celery -A src.worker.celery_app worker -B -l info    # (--pool=solo on Windows)
```

The webhook service then only enqueues; the `worker` service runs the jobs
and writes results to the shared `skynet-data` volume, where `/results`
reads them. The worker keeps diagnosis history in `MLCOP_MEMORY_DB_PATH`
(on the volume in compose).

### Exporting features (live drift data)

A model server publishes the features it sees; Prometheus scrapes them; the
sweep compares them with the baseline:

```python
from src.observability.exporter import observe, serve
serve(9090)                                  # once, at startup
observe("fraud-clf", {"amount": 42.0, "age": 31})   # per prediction
```

Point a scrape job at that port in `prometheus.yml`. To see the drift panels
without a real model, run the *simulated* one (synthetic data, demo only):

```bash
docker compose --profile demo up exporter    # `amount` drifts after DEMO_DRIFT_AFTER=600s
```

with `PROMETHEUS_URL` set (compose does this) and
`MLCOP_MONITOR_INTERVAL_SECONDS=60`, `MLCOP_MONITORED_MODELS=fraud-clf`.
The first sweep records the baseline; drift shows up after the shift.

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

### Demo via Docker

```bash
# Same as `python -m demo.run_demo`, in the demo image (profile: demo).
# Evidently HTML lands in ./demo/out on the host.
docker compose run --rm demo owner/repo 42
docker compose run --rm demo owner/repo 42 --post   # also comment on the real PR

# Dashboard demo with sample data on every panel (no backend needed)
SKYNET_DATA_MODE=sample docker compose up -d dashboard
```

`demo.run_demo` does not write the dashboard's cache. To make the dashboard
show a live PR review, run the worker job inside the webhook container
(needs `GITHUB_TOKEN`, optionally `GROQ_API_KEY`); it writes
`data/prs/<repo>/pr_<n>_<sha>_review.json` to the `skynet-data` volume:

```bash
docker compose up -d webhook dashboard
docker compose exec -T webhook python -c "
from src.core.config import settings
from src.github_integration.pr_client import fetch_pr
from src.worker.jobs import _run_pr_code_review
repo, n = 'owner/repo', 42
_run_pr_code_review(repo, n, fetch_pr(repo, n, settings.github_token).commit_sha)
"
curl -i http://localhost:8000/results/pr-review   # check, then refresh the dashboard
```

Reset the dashboard to "no live data yet" by clearing the volume's cache:

```bash
docker compose exec -T webhook sh -c 'rm -rf /app/data/prs/* /app/data/monitoring/*'
```

On Git Bash, prefix `docker compose exec` paths with `MSYS_NO_PATHCONV=1`
or the `/app/...` path gets rewritten to a Windows one.

If host port 3000 is taken, move the dashboard: `DASHBOARD_PORT=3001 docker compose up -d dashboard`
(or set `DASHBOARD_PORT` in `.env`).

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

It shows live data when the Python side has produced any (latest PR review,
latest monitoring-sweep drift/root cause), and labels any panel still on
sample data. The sweep needs `PROMETHEUS_URL` +
`MLCOP_MONITOR_INTERVAL_SECONDS` > 0 and a model exporting
`mlcop_feature_value` - see "Exporting features" below.

Live data comes over HTTP from the webhook service's read-only results API
when `SKYNET_API_URL` is set, otherwise from its data dir on disk:

```bash
# terminal 1: the Python service (serves /results/pr-review, /results/drift, /results/root-cause)
uvicorn src.api.webhooks:app --port 8000
# terminal 2: the dashboard, reading it over HTTP
cd demo-ui && SKYNET_API_URL=http://localhost:8000 npm run start:prod
```

Dashboard env vars:

| Var | Effect |
|---|---|
| `SKYNET_API_URL` | Base URL of the Python service, e.g. `http://localhost:8000`. Unset: read `../data` (or `SKYNET_DATA_DIR`) from disk instead. |
| `SKYNET_API_TOKEN` | Sent as `Authorization: Bearer …`; must equal the service's `RESULTS_API_TOKEN`. |
| `SKYNET_DATA_MODE` | `auto` (default): live where output exists, labelled sample elsewhere. `live`: never show samples (missing output = "no live data yet"). `sample`: every panel shows the bundled fixture, for demos. |
| `SKYNET_STALE_AFTER_SECONDS` | Monitoring output older than this is flagged Stale (default 7200). PR reviews are never stale. |

```bash
SKYNET_DATA_MODE=sample PORT=3001 npm run start:prod   # demo: sample review with 5 findings
```

A live result that can't be read (unparseable file, backend down, rejected
token) shows as Unavailable, never as a quiet fallback to sample data.

With `docker compose up --build` the dashboard is on `localhost:3000`, reading
the webhook service's results API at `http://webhook:8000`. Set
`RESULTS_API_TOKEN` in `.env` before exposing the webhook service publicly:
unset, the results API is open (the service logs a warning at startup).

## 6. Knowledge graph (`graphify-out/`)

```bash
graphify .                 # full rebuild
graphify . --update        # re-extract only changed files
graphify query "<question>"   # ask about the codebase
```
