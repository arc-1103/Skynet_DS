# Skynet_DS — Current Work Summary
*Reference document for PPT — last updated 2026-09-16*

---

## 1. One-line pitch

An automated GitHub App that reviews Pull Requests for ML/Data-Science-specific
engineering flaws (not generic linting) and separately diagnoses *why* a
deployed model's performance regressed — one diagnostic core, two entry
points: **PR webhook (push)** and **on-demand debug (pull)**.

---

## 2. The problem

Generic PR linters (flake8, pylint, SonarQube) don't understand ML/DS code.
They miss the bugs that actually cost time and money in data science
pipelines:

| Anti-pattern | Why it matters |
|---|---|
| `df.iterrows()` | Orders of magnitude slower than vectorized Pandas ops |
| `model.eval()` without `torch.no_grad()` | Wastes GPU memory building an unneeded autograd graph during inference |
| `.fit()` inside a loop | Unbatched, iterative training instead of proper epoch/batch training |
| Silent data drift in checked-in datasets/fixtures | A PR can quietly reshape a dataset with no readable diff |
| `eval()`, `pickle.loads()`, `shell=True`, hardcoded secrets | Security holes that generic linters under-prioritize in ML codebases |

Separately, ML engineers lose disproportionate time answering one question in
five disguises ("why did this model regress?", "what changed in the data?",
"what caused the metric delta post-deploy?") — Skynet_DS treats these as
**one product** with multiple trigger points, not five separate tools.

---

## 3. Two products, now reconciled into one system

Early in the build there was a fork: the original README describes a
**PR-review bot**; four planning docs (`PRD.md`, `Architecture.md`,
`BluePrint.md`, `Workflow.md`) independently describe an **"ML Reliability
Copilot"** (root-cause diagnosis for model regressions). Both are now live in
the same webhook receiver (`src/api/webhooks.py`), routed by GitHub event
type:

| GitHub event | Product | Handler |
|---|---|---|
| `pull_request` (opened/synchronize) | **PR Review Bot** | `run_pr_code_review` → fetches PR diff, runs drift checks on any changed data files |
| `deployment_status` (success) | **ML Reliability Copilot** | `run_deployment_regression_check` (scheduled 1hr post-deploy) → root-cause diagnosis |

Both share the same underlying primitives: evidence gathering → causal/
statistical analysis → ranked explanation → human-readable output.

---

## 4. Architecture

**Design principle:** one diagnostic core, pluggable data sources, two entry
points (push via webhook, pull via CLI/on-demand).

```
GitHub Webhook ──┐                      Scheduled drift check
CLI query ────────┼──> api/ (FastAPI) <──── worker/ (async dispatch)
                  │
                  v
              agents/ (LangGraph orchestration)
                  │
      ┌───────────┼────────────┬─────────────┐
      v            v            v             v
  security_agent architecture_agent ml_debugger_agent  drift_agent
  (regex/AST)   (regex/AST)      (regex/AST)      (statistical)
      │            │            │             │
      └───────────┴────────────┴─────────────┘
                  v
            synthesizer (fan-in) ──> FinalPRReview (markdown + labels)
```

For the ML Reliability Copilot path:

```
CLI/webhook ──> agents/orchestrator.diagnose()
                  plan → gather (tools/) → hypothesize → rank (graph/CausalGraph)
                  → explain → persist to memory/Store
                  evidence sourced from observability/ (telemetry) + github_integration/ (diffs)
```

### Module responsibilities

| Folder | Responsibility |
|---|---|
| `api/` | Thin HTTP surface — GitHub webhook receiver (FastAPI) |
| `agents/` | LangGraph orchestration — both the multi-agent PR-review graph and the single-run diagnostic state machine |
| `worker/` | Async job dispatch (in-process `asyncio` queue — no Celery/Redis yet, by design) |
| `drift/` | Statistical drift-detection engine (6 methods) + SHAP-based root-cause attribution |
| `graph/` | Causal candidate graph — nodes are candidate causes (code/config/data), ranked by confidence |
| `github_integration/` | GitHub REST API (PR fetch) + local git clone/checkout for diffing real files |
| `observability/` | Telemetry adapters (currently a `FakeTelemetryAdapter`; MLflow/W&B pending a product decision) |
| `memory/` | Storage-agnostic `Store` interface, SQLite implementation — baselines + diagnosis history |
| `retrieval/` + `tools/` | Query layer + LLM-tool-call wrappers used by the orchestrator |
| `schemas/` | Shared Pydantic data contracts — the seam that makes adapters/agents swappable |
| `demo/` | Presentation layer only (see §7) — not part of the core product |

---

## 5. What's actually built (by component)

### 5.1 Multi-agent PR review (`src/agents/`)
Four specialist LangGraph nodes run **in parallel** over a PR's diff, then
fan into one synthesizer:

- **`security_agent.py`** — 8 regex-based checks: `eval()`/`exec()`, unsafe
  `pickle` deserialization, `yaml.load()` without `SafeLoader`,
  `subprocess(shell=True)`, `os.system()`, disabled TLS verification,
  hardcoded credentials.
- **`architecture_agent.py`** — structural smells: bare `except:`, `print()`
  instead of logging, deeply nested logic, oversized diffs (>300 added
  lines in one file).
- **`ml_debugger_agent.py`** — the project's original purpose: `.iterrows()`,
  missing `torch.no_grad()` around `.eval()`, `.fit()` called inside a loop.
- **`drift_agent.py`** — wraps the statistical drift engine as a graph node;
  only runs when a local repo checkout is available.
- **`synthesizer.py`** — merges all four outputs into one `FinalPRReview`:
  markdown summary, PR labels (`security`/`performance`/`refactor`/`config`),
  and a pass/fail verdict (blocks on HIGH/CRITICAL severity findings).
- **`diff_utils.py`** — shared unified-diff parser; every finding is anchored
  to a real added line number so it can become a GitHub inline comment.

All four specialists are **deterministic (regex/statistics), not LLM calls**
— a documented, deliberate choice (no LLM provider/API key wired into this
environment yet). The node interface is identical to what an LLM-backed
version would use, so swapping one specialist for a real Claude/Groq/Gemini
call later is a single-file change.

### 5.2 Data drift engine (`src/drift/`)
- **`methods.py`** — 6 independently-implemented drift detectors (no
  black-box dependency on Evidently for the actual pass/fail decision):
  Kolmogorov-Smirnov, Population Stability Index, Page-Hinkley (sequential
  change-point detection), Euclidean distance, Cosine distance, and a
  model-based domain classifier (RandomForest + ROC AUC).
- **`detector.py`** — runs all applicable detectors per column, rolls up to
  a dataset verdict via **"Share of Drifted Components"** with majority-vote
  or any-vote aggregation (deliberately not "any column drifted" — avoids
  false positives on wide tables).
- **`root_cause.py`** — **SHAP-based root-cause attribution**: trains an
  XGBoost surrogate to distinguish reference vs. current rows, then reads
  `shap.TreeExplainer` to rank *which features* actually explain the drift
  (with statistical gating: AUC must beat a threshold **and** be
  significant via Mann-Whitney U, or the ranking is suppressed as noise).
  Feeds directly into `graph/CausalGraph` as ranked `CauseCandidate`s.
- **`pr_drift.py`** — applies the above to a PR: diffs every changed
  `.csv`/`.parquet` file's base-branch vs. head-branch version via
  `git show`, so a quietly-reshaped dataset shows up as per-column drift
  instead of an unreadable multi-thousand-row diff.

### 5.3 ML Reliability Copilot core (`src/agents/orchestrator.py` + support)
- End-to-end diagnostic pipeline: plan → gather evidence (`tools/`) →
  hypothesize (config/data/code candidates) → rank (`graph/CausalGraph`) →
  explain → persist verdict (`memory/Store`).
- `src/cli.py` — `python -m src.cli debug --run-id <id>` (the "pull" entry
  point).
- Runs end-to-end today against a `FakeTelemetryAdapter` (seeded regression:
  a `learning_rate` bump between two runs, accuracy 0.91→0.83) — proves the
  pipeline shape; a real MLflow/W&B adapter is a pending product decision.

### 5.4 GitHub integration (`src/github_integration/`)
- `pr_client.py` — fetches PR metadata + per-file diffs via the GitHub REST
  API (PyGithub), normalized into `schemas/pr.py`.
- `repo_client.py` — clones/checks out a repo locally via the system `git`
  CLI so drift detection can read real file contents, not just diff patches.

### 5.5 Webhook + async worker (`src/api/webhooks.py`, `src/worker/`)
- FastAPI receiver, HMAC-SHA256 signature verification
  (`X-Hub-Signature-256`), routes by `X-GitHub-Event`.
- `pull_request` (opened/synchronize) → caches the PR locally + runs drift
  checks on any changed data files.
- `deployment_status` (success) → schedules a **1-hour delayed** regression
  check via an in-process `DelayedTaskQueue` (intentionally not
  Celery/Redis — avoids infra overbuild before the diagnostic core is
  validated).

### 5.6 Demo/presentation layer (`demo/`) — not part of the core product
Answers "where is the UI?" by pointing at tools that already have one,
rather than building a UI from scratch:
1. **GitHub itself** (`github_pr_comment.py`) — posts the `FinalPRReview` as
   a real GitHub PR review: a summary comment + one inline comment per
   line-anchored finding with a suggested fix.
2. **Evidently's dashboard** (`evidently_dashboard.py`) — renders the same
   reference/current data comparisons as a standalone interactive HTML
   report (presentation-only; the actual pass/fail decision still comes
   from `src/drift/`).
3. **Langfuse tracing** (`langfuse_tracing.py`) — visualizes the four
   specialist agents executing concurrently inside the LangGraph.
   *(Honest gap: since agents are currently deterministic, not LLM-backed,
   the trace shows concurrency/latency but not token/model metrics yet.)*

---

## 6. Current status

- **213 tests passing** across 25 test files (unit + integration, including
  a `fastapi.testclient` suite for the webhook and an end-to-end diagnose
  workflow test).
- Two real bugs caught by tests and fixed during development: a diagnosis
  history ordering bug (SQLite second-resolution timestamps) and a malformed
  webhook payload returning a raw 500 instead of a clean 422.
- Both entry points (PR-review webhook, deployment-regression webhook) are
  wired and tested end-to-end against fake/deterministic data.

### Deliberately not yet done (open decisions, not oversights)
- **No live LLM call anywhere yet** — every "agent" is regex/statistics
  based. The interface is LLM-call-ready; wiring one in is scoped to a
  single file per agent.
- **No real telemetry adapter** — MLflow vs. Weights & Biases is an open
  product decision; only a fake adapter exists today.
- **The multi-agent PR-review graph isn't yet posting back to GitHub
  automatically** — the webhook currently caches the PR + runs drift checks;
  posting the full `FinalPRReview` as a real GitHub comment is demonstrated
  in `demo/github_pr_comment.py` but not yet wired into the production
  webhook path.
- **No durable job queue** — the async worker is in-memory only; a scheduled
  check is lost if the process restarts within its delay window (acceptable
  for a single-process v1 devtool, flagged for later).
- **No labeled eval set** for measuring root-cause precision against ground
  truth (PRD success-metric target: ≥60% top-1 accuracy) — not built yet.

---

## 7. Tech stack

| Layer | Choice |
|---|---|
| Agent orchestration | LangGraph (multi-agent PR review) + hand-rolled state machine (single-run diagnosis) |
| API | FastAPI |
| Async jobs | Plain `asyncio` task queue (no Celery/Redis by design, v1) |
| Storage | SQLite behind a storage-agnostic `Store` interface |
| Drift detection | NumPy/SciPy/scikit-learn (custom implementations, not a black-box library) |
| Root-cause attribution | XGBoost + SHAP |
| GitHub integration | PyGithub (API) + system `git` (local clone/diff) |
| Data contracts | Pydantic v2 schemas throughout |
| Demo-only | Evidently (dashboard), Langfuse (tracing) |
| Testing | pytest, `fastapi.testclient` — 213 tests |

---

## 8. Suggested PPT structure

1. **Title / one-line pitch** (§1)
2. **Problem** — why generic linters miss ML/DS bugs (§2, table)
3. **Product shape** — one core, two entry points (push webhook / pull CLI) (§3)
4. **Architecture diagram** (§4 — redraw the ASCII diagrams as a clean visual)
5. **Multi-agent PR review demo** — 4 specialists → synthesizer → GitHub
   comment (§5.1, §5.6.1) — best slide for a live/screenshot demo
6. **Drift + root-cause engine** — the statistical/SHAP differentiator (§5.2)
7. **ML Reliability Copilot** — root-cause diagnosis workflow (§5.3)
8. **Status & proof points** — 213 tests passing, real bugs caught by tests (§6)
9. **What's next** — LLM-backed agents, real telemetry adapter, GitHub
   auto-posting, labeled eval set (§6, gaps list)
10. **Tech stack** (§7)
