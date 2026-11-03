# Skynet_DS: Report Context

*A single reference for what the project is, what it does today, what makes it
different, and where it is going. Compiled 2026-11-03 from the source tree and
the planning docs (`PRD.md`, `Architecture.md`, `Workflow.md`, `PRODUCT.md`,
`PRESENTATION.md`, `HOW_TO_RUN.md`). `current_work.md` (2026-09-16) predates
several features below and is superseded by this file.*

---

## 1. TL;DR

**Skynet_DS ("ML Reliability Copilot") answers one question: given an ML
model or ML codebase that is behaving worse than expected, *why*?**

It does that from two directions with one diagnostic core:

- **Before merge (push).** A GitHub App webhook reviews every pull request for
  *ML/data-science-specific* flaws: vectorization failures, GPU-memory waste,
  data leakage, unsafe deserialization, silently reshaped datasets. It can open
  a second PR that applies the safe, mechanical fixes.
- **After deploy (push + pull).** When a deployed model regresses or drifts,
  it diffs code, config, data and metrics against a baseline, ranks the causes
  with confidence scores and evidence, and explains them in plain language,
  via CLI, webhook, a continuous monitoring sweep, Slack and a dashboard.

The core idea: **diagnose, don't just detect.** Most tools stop at "something
drifted" or "this line looks risky". Skynet_DS goes on to "this feature
explains the drift, and this commit/config change most likely caused it."

---

## 2. The Problem

### 2.1 Generic code review does not understand ML code

| Anti-pattern | Why it hurts | Generic linters |
|---|---|---|
| `df.iterrows()` in a hot path | Much slower than vectorized pandas | Not flagged |
| `model.eval()` without `torch.no_grad()` | Builds an unneeded autograd graph and wastes GPU memory | Not flagged |
| `.fit()` inside a loop | Unbatched, repeated training | Not flagged |
| Scaler fitted before the train/test split | Data leakage; inflated offline metrics | Not flagged |
| A PR quietly reshapes a `.csv`/`.parquet` fixture | The diff is thousands of unreadable rows | Invisible |
| `pickle.loads`, `yaml.load`, `verify=False`, hardcoded keys | Real security holes, common in ML notebooks-turned-services | Often low priority |

### 2.2 "Why did my model get worse?" is one question in five disguises

| Disguise | Underlying question |
|---|---|
| ML Debugger | Why is this model wrong right now? |
| Drift → Root Cause | What changed in the data or pipeline? |
| Experiment Intelligence | Across my runs, which change caused the delta? |
| Deployment Optimizer | What drove the post-deploy metric drop? |
| Reliability Copilot | Is something wrong right now, and why? |

All five need the same primitives: **baseline → current state → diff across
code / config / data / metrics → ranked causal explanation.** Skynet_DS
builds those primitives once instead of shipping five tools.

---

## 3. Product Shape: One Core, Many Triggers

```
 GitHub webhook (pull_request) ─┐
 GitHub webhook (deployment)  ──┤                     ┌─ Security agent ──────┐
 CLI: debug / compare / drift ──┼─► api/ ─► worker/ ─►├─ Architecture agent ──┼─► Synthesizer ─► FinalPRReview
 Monitoring sweep (scheduled) ──┘   (FastAPI) (asyncio│─ ML-Debugger agent ───┤        │            │
                                     or Celery+Redis) └─ Drift agent ─────────┘        │            ├─► fix PR (opt-in)
                                          │                                            │            ├─► /results API → dashboard
                                          ▼                                            │            └─► GitHub review (demo)
                         agents/orchestrator: plan → gather → hypothesize → rank → explain
                                          │
          tools/ → retrieval/ → observability/ (Fake · W&B · Prometheus) + github_integration/ + memory/ (SQLite)
                                          │
                              graph/CausalGraph (ranked CauseCandidates + history prior)
                                          │
                              drift/ (7 detectors + SHAP root cause) → Slack / dashboard / CLI
```

**Design rules that hold the system together**

- **Adapter boundary.** Telemetry backends translate *into* `schemas/`.
  Nothing in `agents/`, `graph/` or `worker/` imports W&B or Prometheus, so
  adding a backend means adding one file in `observability/`.
- **Thin API.** `api/` routes and validates; business logic lives in
  `agents/`, `drift/` and `graph/`.
- **Workflow 5 schedules, it doesn't diagnose.** The monitoring sweep reuses
  Workflows 2 and 4 and adds no diagnostic logic of its own.

---

## 4. Feature Inventory (what exists today)

### 4.1 Multi-agent PR review (`src/agents/`, LangGraph)

Four specialists run **in parallel** over the PR diff, then a synthesizer
fans them in. Each branch is isolated, so one failing agent cannot crash the
review.

| Agent | Deterministic checks (always on) | LLM pass (when `GROQ_API_KEY` set) |
|---|---|---|
| **Security** | `eval`/`exec`, unsafe `pickle`, `yaml.load` without SafeLoader, `shell=True`, `os.system`, disabled TLS, hardcoded credentials, high-entropy secrets, dynamic SQL (AST-based tools) | Tool-calling loop: secret scan, AST security scan. Finds SSRF, path traversal, etc. |
| **Architecture** | Bare `except:`, `print` instead of logging, deep nesting, oversized diffs (>300 added lines/file) | Structural issues no pattern covers |
| **ML-Debugger** | `.iterrows()`, `.eval()` without `no_grad`, `.fit()` inside a loop | Tools: dataset schema inspection, drift analysis, SHAP root cause, deprecated-API doc check. Finds leakage, missing seeds, chained assignment, unbatched `.apply()`, GPU metric without `.item()` |
| **Drift** | Statistical drift on every changed tabular data file (base vs. head via `git show`) | n/a |
| **Synthesizer** | Merges all findings → `FinalPRReview`: markdown summary, labels (`security`/`performance`/`refactor`/`config`), pass/fail verdict (blocks on HIGH/CRITICAL) | n/a |

Every finding is anchored to a real added line (`diff_utils.py`) so it can
become a GitHub inline comment, and duplicates across agents are removed.

### 4.2 Autonomous fix PRs (`src/github_integration/fix_pr.py`)

After a review, the agent can open a **second PR into the author's branch**
that applies exact one-line patches:

- `yaml.load(x)` → `yaml.safe_load(x)`
- `verify=False` → `verify=True`
- bare `except:` → `except Exception:`

Safeguards:
- Off by default. `MLCOP_OPEN_FIX_PRS=dry-run` logs the patch and writes nothing.
- A line is patched only if it still matches exactly what was reviewed.
- Never targets the default branch and never merges.
- A new push closes the old fix PR as "Superseded" and deletes its branch.
- Fork PRs, closed PRs and the agent's own fix PRs are skipped, so it never fixes its own fixes.
- Half-written branches are cleaned up, and a fix-PR failure never fails the review.
- `eval`, `shell=True`, pickle and **every LLM suggestion stay comment-only**.

This flow has been verified live on a public scratch repo.

### 4.3 Drift detection engine (`src/drift/`)

Custom implementations on NumPy/SciPy/scikit-learn. Evidently is used only for
presentation, not for the decision.

| Method | Use |
|---|---|
| Kolmogorov–Smirnov | Numeric distribution shift |
| PSI (Population Stability Index) | Binned shift, industry standard |
| Page-Hinkley | Sequential change-point detection (flags unstable baselines) |
| Euclidean / Cosine distance | Binned / vector distribution distance |
| MMD (RBF kernel) | Spread/shape changes even when the mean is unchanged |
| Model-based (domain classifier) | Can a classifier tell reference from current rows? |

Dataset verdict uses **"Share of Drifted Components"** with majority-vote or
any-vote aggregation, not "any column drifted", which avoids false positives
on wide tables. Mixed-type and categorical columns are handled explicitly.

### 4.4 SHAP root-cause attribution (`src/drift/root_cause.py`)

Drift tests say *whether* a column moved. This engine says *which features
account for it*:

1. Train an **XGBoost surrogate** to separate reference rows from production rows.
2. Rank features with **SHAP TreeExplainer**.
3. **Statistical gating**: the surrogate's AUC must beat a threshold *and* be
   significant (Mann-Whitney U), or the result is reported as noise.
4. **Held-out rows only**, so a memorizing tree on small, wide data cannot
   fake a confident cause.
5. Labels each shift as a change in **level** or in **shape**.
6. Emits ranked `CauseCandidate`s with discounted confidence into `graph/`.

Because the surrogate sees columns **jointly**, it discounts features that only
look shifted because they correlate with the real culprit. Per-column tests
can't do that.

### 4.5 ML Reliability Copilot: the diagnostic core (`src/agents/orchestrator.py`)

The pipeline is **plan → gather evidence → hypothesize (code / config / data)
→ rank (`CausalGraph`) → explain → persist**. It implements all five workflows
from `Workflow.md`:

| # | Workflow | Entry point |
|---|---|---|
| 1 | On-demand debug | `python -m src.cli debug --run-id <id>` |
| 2 | Drift detection + explanation | `python -m src.cli drift --model-id <m>`, worker job, episode-aware (one diagnosis per drift *episode*, not per check) |
| 3 | Experiment comparison (N-way) | `python -m src.cli compare --runs A,B,C --metric <m>` |
| 4 | Deployment regression triage | `deployment_status` webhook → check scheduled 1 h later, scoped to the deploy's commit range |
| 5 | Continuous monitoring | Scheduled sweep over watched models (`MLCOP_MONITOR_INTERVAL_SECONDS`) |

Other pieces of the core:
- **Causal ranking.** Candidates are typed (code · config · data · unknown) and carry a 0–1 confidence plus evidence references.
- **History prior.** Past verdicts in SQLite nudge near-ties toward this model's usual culprit, capped at about ±30% so clear evidence still wins.
- **Explanations.** Deterministic by default. An optional LLM explanation (`MLCOP_LLM_EXPLAIN`, `prompts/explain.md`) is a separate opt-in because it sends run data to Groq.

### 4.6 Data sources and integrations

| Area | What exists |
|---|---|
| Telemetry adapters | `FakeTelemetryAdapter` (seeded regression), `WandbTelemetryAdapter` (real), `PrometheusAdapter` (live feature distributions) |
| Feature exporter | `observability/exporter.py`: `observe(model_id, features)` + `serve(port)` publish `mlcop_feature_value` for Prometheus; a simulated drifting exporter for demos |
| GitHub | PyGithub for PR/commit fetch; system `git` for clone/checkout with credentials passed as a header |
| Code indexing | Tree-sitter AST parsing + Python-DS-aware chunker (`indexing/`) to scope evidence to the enclosing function/class |
| Memory | Storage-agnostic `Store` protocol with a SQLite implementation: baselines, diagnosis history, active drift episodes |
| Notifications | Slack incoming webhook for drift and deploy-regression verdicts |

### 4.7 Runtime and delivery

| Component | Details |
|---|---|
| Webhook service | FastAPI; HMAC-SHA256 signature verification; data-driven event router; `/healthz` |
| Worker | In-process asyncio delayed-task queue by default; **Celery + Redis** durable worker optional (`CELERY_BROKER_URL`), with per-model fan-out so one slow SHAP fit can't block others |
| Results API | Read-only `/results/pr-review`, `/results/drift`, `/results/root-cause`, behind a bearer token |
| Dashboard (`demo-ui/`) | NestJS + vanilla HTML/CSS/JS, dark-first with light toggle. Verdict + KPIs, findings table, detector matrix, drift + SHAP attribution panels. **Never shows sample data as live**: every panel is labeled live / sample / stale / unavailable |
| Demo layer (`demo/`) | Post the review as a real GitHub review with inline suggestions; Evidently HTML drift dashboards; Langfuse traces showing the four agents running concurrently |
| Deployment | Docker Compose: webhook, dashboard, Prometheus; profiles for CLI, demo, exporter, and Redis + worker. Non-root images, healthchecks, persistent volumes |

### 4.8 Quality and evaluation

- **426 test functions in 43 files** (`tests/` + `demo/tests/`), more cases once
  parametrization expands. They cover agents, all drift methods, root cause,
  API, worker, Celery path, fix PRs, adapters and LLM client fakes. LLM calls
  are disabled by default in tests.
- **Labeled-incident eval** (`python -m evals.run_eval`), with two gates from
  the PRD: top-1 cause precision **≥ 60%** and time-to-root-cause **< 2 min**.
  The set has **20 incidents, all synthetic**. `PRESENTATION.md` reports
  **80% top-1**. All four misses share one pattern: an incidental config tweak
  outranks the real code or data cause.
- `evals/add_incident.py` captures real regressions with their true cause. Real
  incidents start gating once 5 or more exist, and precision is reported per
  source.

### 4.9 Security hardening of the LLM tool surface

Since the LLM has tools, its tool surface is treated as an attack surface:

- **SSRF guard** on any URL the model asks to fetch, including redirect-walking.
- **Repo-path containment**: tools cannot read outside the PR checkout.
- **Commit SHA validation** before a SHA reaches a file path or URL.
- Git credentials sent via HTTP header, never embedded in URLs.
- Webhook signature verification (forged signature → 401, verified live).
- Results API bearer token (401 without it, tested from outside).
- Config typos fail safe: an unrecognized `MLCOP_OPEN_FIX_PRS` value means "off".

---

## 5. Innovations

1. **One diagnostic core for five problems.** PR review, drift explanation,
   experiment comparison, deploy triage and continuous monitoring share one
   evidence → causal graph → explanation pipeline instead of being separate
   products.
2. **Pre-merge and post-deploy in one system.** The same tool that flags
   `.iterrows()` in a PR also explains why the deployed model drifted last
   week, with shared schemas and shared memory.
3. **Drift *in pull requests*.** Changed `.csv`/`.parquet` files are diffed
   statistically (base vs. head), so a silently reshaped dataset becomes a
   readable per-column drift report.
4. **Root cause, not just anomaly.** The surrogate + SHAP pipeline ranks *which
   feature* explains a drift, gated by separability and significance tests,
   on held-out rows. When the signal is weak it says so instead of inventing a
   culprit.
5. **Typed causal ranking with memory.** Candidates are code / config / data
   with confidence and evidence, and a capped history prior learns each model's
   usual failure modes without overriding clear evidence.
6. **Deterministic first, LLM second, fails open.** The product is fully useful
   at zero LLM spend. The LLM adds coverage through a real tool-calling loop and
   is never a single point of failure.
7. **Trust-bounded auto-remediation.** The system writes code only for exact,
   deterministic, re-verified one-line patches. Anything an LLM suggested is
   never committed. It opens a PR into the author's branch, supersedes stale
   fix PRs, and never merges.
8. **Detector ensemble with roll-up.** Seven independently implemented
   detectors with "Share of Drifted Components" aggregation, instead of one
   black-box test.
9. **Honest presentation.** The dashboard labels live, sample, stale and
   unavailable data explicitly, and the eval reports misses and synthetic
   provenance openly.

---

## 6. What We Do Differently

Comparison by **category**, not by named product, because individual tools
change quickly.

| Category | What they typically do | What Skynet_DS does differently |
|---|---|---|
| **Generic linters / static analysis** | Style, syntax, general bug patterns, language-agnostic security rules | ML/DS-specific checks (vectorization, GPU context, unbatched training, leakage); drift on data files in the PR; line-anchored findings + safe auto-fix PRs |
| **General AI PR reviewers** | LLM comments on any diff | Deterministic layer first (free, tested, reproducible); LLM pass has domain tools (schema inspection, drift, SHAP); hardened tool surface; LLM output is never auto-committed |
| **Drift / ML monitoring platforms** | Detect and alert that distributions or metrics shifted | Explain *why*: SHAP surrogate attribution plus correlation with code/config/pipeline commits, ranked by confidence; episode-aware so one drift produces one diagnosis |
| **Experiment trackers** | Store runs, metrics and configs; visual comparison | Reads from them (W&B adapter) instead of replacing them, and ranks which change caused the metric delta across N runs |
| **Observability stacks (metrics/dashboards)** | Time series and alert thresholds | Uses Prometheus as an evidence source, then runs the diagnostic pipeline on what it finds |

**The short version:** others detect, Skynet_DS diagnoses, and it does so on
both sides of the merge button.

Explicit **non-goals** (from `PRD.md`) keep the scope sharp: it is not an
experiment tracker, not an observability platform, no auto-rollback or
auto-retraining in v1, and no multi-tenant auth or billing in v1.

---

## 7. Current Maturity (honest status)

| Area | Status |
|---|---|
| Multi-agent PR review | Working; deterministic + optional LLM pass |
| Fix PRs | Working, verified live on a scratch repo; 3 patterns only |
| Drift engine + SHAP root cause | Working, heavily tested |
| Diagnostic core (Workflows 1–5) | Working end-to-end; Fake + W&B + Prometheus adapters |
| Celery + Redis | Unit-tested; **not yet run against a real Redis** |
| Exporter → Prometheus → sweep | Library exists; **no real model publishes through it yet**, so the dashboard shows labeled sample/synthetic data |
| Deployment | Docker Compose locally; needs a tunnel (e.g. ngrok) for GitHub to reach it; **not permanently hosted** |
| LLM pass | Groq (Qwen) free tier throttled to ~1000 output tokens/min; reviews still complete on deterministic checks |
| Eval | 20 synthetic incidents; 80% top-1 reported; config-vs-real-cause miss pattern not yet tuned |
| Identity | Fix-PR commits are authored by the token owner, not an app identity |

---

## 8. Future Projections

### 8.1 Near term (committed roadmap, from `PRESENTATION.md` / `PRD.md`)

- **Real incidents in the eval set** so the ≥60% precision gate measures
  something real (`evals/add_incident.py`).
- **Confirmed-verdict feedback.** Today verdicts are saved whether or not they
  were right, so a wrong pick can reinforce itself. The history prior should
  learn from human-confirmed answers only.
- **Down-weight incidental config tweaks**, which account for all four current
  eval misses.
- **GitHub App identity** with only contents + pull-requests write; record
  fix-PR outcomes for a dashboard activity panel.
- **Permanent hosting**, so no tunnel is needed.
- **Validate the durable queue** against real Redis.
- **First real exporter integration**, so live drift panels show real traffic.

### 8.2 Mid term (natural extensions the architecture already allows)

*Proposals, not commitments. Each one plugs into an existing seam.*

- **More fix patterns**: further deterministic, exactly verifiable rewrites,
  e.g. wrapping inference in `torch.no_grad()` where the scope is unambiguous.
- **MLflow adapter**, the other v1 candidate in `Architecture.md`. It is one
  file in `observability/`.
- **Team dashboard**: history of reviews, drift episodes and diagnoses per
  model, read from `memory/`.
- **More notification sinks** (Teams, email, PagerDuty) next to Slack.
- **Infra/hardware causes** in the causal graph (GPU type, library versions,
  container image). The PRD explicitly scopes these out of v1.
- **Swap SQLite for a server DB** behind the existing `Store` protocol.

### 8.3 Long term (product direction)

- **SaaS / multi-tenant** (PRD G4: "devtool-first, SaaS-ready later"): tenant
  isolation in `memory/`, auth, billing.
- **Action agents**: with confirmed-verdict history and a strong precision
  record, move from "diagnose" toward human-approved remediation such as
  rollback suggestions or retraining triggers (PRD NG3 defers this beyond v1).
- **Learned ranking**: once enough labeled incidents exist, replace the
  hand-tuned prior with a model trained on per-model-family cause patterns
  ("`learning_rate` changes caused 70% of this family's regressions").
- **Split entry points into products only if the market asks.** The PRD
  deliberately keeps one product until one entry point shows its own demand.

---

## 9. Tech Stack

| Layer | Tools |
|---|---|
| Orchestration | LangGraph, Groq (Qwen models), Instructor |
| API / worker | FastAPI, Uvicorn, asyncio queue, Celery + Redis (optional) |
| ML / statistics | NumPy, SciPy, pandas, scikit-learn, XGBoost, SHAP |
| Code analysis | Tree-sitter, Python `ast`, unidiff, PyGithub, system `git` |
| Observability | Prometheus, Weights & Biases, Langfuse, Evidently |
| Storage | SQLite behind a `Store` protocol |
| Contracts | Pydantic v2 schemas throughout |
| Frontend | NestJS + vanilla HTML/CSS/JS (no build step for the page itself) |
| Delivery | Docker Compose with profiles; non-root images; healthchecks |
| Testing | pytest, FastAPI TestClient, fake adapters/clients |

---

## 10. Key Configuration

| Variable | Gates |
|---|---|
| `GITHUB_TOKEN` | PR fetch, review pipeline, fix PRs (needs contents + PR write) |
| `GITHUB_WEBHOOK_SECRET` | Webhook signature verification |
| `GROQ_API_KEY` / `GROQ_MODEL` | LLM pass in specialist agents (unset = deterministic only) |
| `MLCOP_LLM_EXPLAIN` | LLM-written diagnosis explanations (separate opt-in) |
| `TELEMETRY_BACKEND` | `fake` (default) or `wandb` |
| `PROMETHEUS_URL` | Live drift checks and the monitoring sweep |
| `MLCOP_MONITOR_INTERVAL_SECONDS` / `MLCOP_MONITORED_MODELS` | Workflow 5 schedule and scope |
| `MLCOP_REPO_MODELS` / `MLCOP_PIPELINE_PATHS` / `GITHUB_REPO` | Map repos to models; pipeline-change evidence; code-diff source |
| `MLCOP_OPEN_FIX_PRS` | `off` / `dry-run` / `on` |
| `CELERY_BROKER_URL` | Durable queue |
| `RESULTS_API_TOKEN` | Locks `/results` (set before exposing the service) |
| `SLACK_WEBHOOK_URL` | Verdict alerts |

---

## 11. Where to Look

| Need | File |
|---|---|
| Problem, goals, success metrics | `PRD.md` |
| Module responsibilities, adapter boundary | `Architecture.md` |
| The five workflows step by step | `Workflow.md` |
| Dashboard purpose and principles | `PRODUCT.md`, `DESIGN.md` |
| Slide deck (Marp) | `PRESENTATION.md` |
| Every run command | `HOW_TO_RUN.md` |
| Codebase map (knowledge graph) | `graphify-out/GRAPH_REPORT.md`, `graphify-out/graph.html` |
| Core abstractions (most connected) | `NormalizedRun`, `SQLiteStore`, `EvidenceRetriever`, `PRPayload`, `ChangedFile`, `CauseCandidate`, `explain_drift()` |
