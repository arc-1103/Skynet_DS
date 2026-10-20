---
marp: true
theme: default
paginate: true
title: Skynet_DS - ML Reliability Copilot
---

<!-- Slides are separated by `---`. Text inside HTML comments is speaker notes.
     Open in VS Code with the "Marp for VS Code" extension and export to .pptx,
     or paste into Gamma / Google Slides one section per slide. -->

# Skynet_DS
### ML Reliability Copilot

**Given a model behaving worse than expected, explain *why*.**

Multi-agent PR review for ML code + drift detection with root-cause attribution

<!-- Open with the one-line question. Everything else in the deck serves it. -->

---

## The Problem

ML engineers lose hours answering one question in five disguises:

| Disguise | Underlying question |
|---|---|
| ML Debugger | Why is this model wrong right now? |
| Drift → Root Cause | What changed in the data/pipeline? |
| Experiment Intelligence | Which change caused the regression? |
| Deployment Optimizer | What drove the post-deploy metric delta? |
| Reliability Copilot | Is something wrong right now, and why? |

<!-- Generic linters catch style, not iterrows() in a hot loop, pickle.load on
     untrusted data, or a feature whose distribution silently shifted. -->

---

## Our Answer: One Core, Two Triggers

- **Push:** GitHub webhook on a PR / deployment → async worker → diagnosis
- **Pull:** CLI on demand → *"why did run X regress?"*
- Both converge on the **same diagnostic core**, not two products
- Output is a **ranked cause with evidence**, not just an anomaly flag

All five use cases need the same primitives:
**baseline → current state → diff (code / data / config / metrics) → causal explanation**

---

## Architecture

```
 GitHub Webhook ─┐                       ┌─ Security agent ──────┐
 CLI (debug /    ├─► api/ ─► worker/ ─►  ├─ Architecture agent ──┼─► Synthesizer ─► FinalPRReview
   compare/drift)│                       ├─ ML-Debugger agent ───┤
 Monitoring sweep┘                       └─ Drift agent ─────────┘
                                                   │
           observability/ adapters (W&B · Prometheus · Fake) ─► schemas/ (normalized)
```

- **FastAPI** webhook service · **Celery + Redis** durable worker (optional; default is an in-process asyncio queue) · **SQLite** memory store
- Adapters normalize telemetry into shared schemas, so nothing downstream imports MLflow or Prometheus directly

<!-- The adapter boundary is what makes swapping W&B for Prometheus cheap. -->

---

## Multi-Agent PR Review (LangGraph)

Four specialists run **in parallel**, then fan in to one synthesizer:

| Agent | Catches |
|---|---|
| **Security** | Unsafe deserialization, shell injection, disabled TLS, hardcoded secrets |
| **Architecture** | Bare `except`, print-logging, deep nesting, oversized diffs |
| **ML-Debugger** | `.iterrows()`, unbatched training, missing `torch.no_grad()`, leakage, missing seeds |
| **Drift** | Statistical drift in any data files the PR changes |

- Each branch is **isolated**: one failing agent can't crash the review
- Result is posted back as a GitHub review with inline suggestions

---

## Two-Pass Design: Deterministic First, LLM Second

1. **Regex / AST checks always run**: free, instant, fully tested
2. **LLM pass (Groq, Qwen models)** runs only when an API key is set
   - Real **tool-calling loop**: the model can scan for secrets, run an AST security scan, inspect dataset schemas, run drift analysis, and call SHAP
   - Finds what no fixed pattern covers: SSRF, path traversal, train/test leakage
3. **Fails open**: no key, SDK error, or bad output simply means no extra findings

<!-- Key design choice: the product is useful with zero LLM spend, and the LLM
     adds coverage without being a single point of failure. -->

---

## Drift Detection Engine

Per-column and dataset-level drift, built on numpy / scipy:

| Method | Use |
|---|---|
| **Kolmogorov–Smirnov test** | Numeric distribution shift |
| **PSI** (Population Stability Index) | Binned shift, industry standard |
| **Page-Hinkley** | Streaming change-point detection |
| **Euclidean / Cosine distance** | Embedding & vector drift |
| **MMD** (RBF kernel) | Multivariate distribution shift |
| **Model-based** | Classifier separability of ref vs. current |

Reports `n_drifted`, `share_drifted`, and a dataset-level drift verdict

---

## Root Cause: *Which* Feature Moved?

Drift tests say **whether** a column moved. The SHAP engine says **which features account for it**.

1. Train an **XGBoost surrogate** to tell reference rows from production rows
2. Read it with **SHAP TreeExplainer** to rank the features it relied on
3. Safeguards built in:
   - **Separability gate**: if the classifier is at chance, attribution is reported as noise
   - **Held-out rows only**: avoids confident attributions from memorized data

Why a surrogate? It sees columns **jointly**, so it discounts features that only look shifted because they correlate with the real culprit

---

## Causal Ranking

- Candidate causes are typed: **code · config · data · unknown**
- Each carries a **confidence score (0–1)** plus evidence references
- Evidence is pulled from run metrics, code diffs (via Tree-sitter AST chunking), and data versions
- Past diagnoses are stored in SQLite memory; a **history prior** nudges near-ties toward this model's usual culprit (capped at roughly ±30%, so clear evidence still wins)

<!-- Honest note if asked: the prior is a tie-breaker. Verdicts are saved whether
     or not they were right, so a wrong pick can reinforce itself until a human
     confirmation signal exists. -->

---

## Three Ways to See the Output

1. **GitHub PR review**: summary comment with severity badges + inline suggested fixes
2. **Dashboard** (NestJS + vanilla JS): verdict & KPIs, findings table, drift + attribution panels, clearly labelled **live vs. sample** data
3. **Observability**: **Langfuse** traces show the four agents running concurrently; **Evidently** HTML drift dashboards; **Prometheus** metrics

<!-- Principle on the dashboard: never present sample data as live. -->

---

## Quality & Evaluation

- **375 automated tests** across agents, drift methods, API, worker, and integrations
- **Labeled-incident eval** (`python -m evals.run_eval`)
  - Gate: **top-1 cause precision ≥ 60%**
  - Gate: **time-to-root-cause < 2 minutes**
- Currently **80% top-1 over 20 synthetic incidents**; precision is reported per source, and real incidents gate once 5 or more are collected
- Known weakness: all 4 misses are the same pattern, an incidental config tweak outranking the real code or data cause

<!-- Be upfront: all 20 incidents are synthetic; the plan is to add real
     regressions with evals/add_incident.py. -->

---

## Security Hardening

The LLM has tools, so its tool surface is treated as an attack surface:

- **SSRF guard** on any URL the model asks to fetch
- **Repo-path containment**: tools can't read outside the PR checkout
- **Commit SHA validation** on webhook payloads before they reach a file path
- Git credentials passed via **header**, never embedded in URLs
- GitHub webhook **signature verification**

---

## Tech Stack

| Layer | Tools |
|---|---|
| Orchestration | LangGraph, Groq (Qwen), Instructor |
| API / Worker | FastAPI, Uvicorn, Celery + Redis (optional), asyncio queue |
| ML / Stats | NumPy, SciPy, pandas, scikit-learn, XGBoost, SHAP |
| Code analysis | Tree-sitter, Python `ast`, unidiff, PyGithub |
| Observability | Langfuse, Prometheus, Evidently, Weights & Biases |
| Frontend | NestJS + vanilla HTML/CSS/JS |
| Deploy | Docker Compose (webhook, dashboard, prometheus, cli, demo) |

---

## Demo Flow

```bash
docker compose up                                   # full stack
python -m src.cli debug --run-id run-004            # why did this run regress?
python -m src.cli compare --runs run-baseline,run-004 --metric eval_accuracy
python -m evals.run_eval                            # precision & latency gate
```

1. Open a PR with an ML anti-pattern + a changed CSV
2. Watch the 4 agents run in Langfuse
3. See the review on GitHub and the drift + SHAP cause on the dashboard

---

## Limitations & Roadmap

**Today**
- Exporter library exists (`observe()`), but no real model publishes through it yet; the dashboard shows labelled sample data until one does
- Eval set is small and entirely synthetic; the config-vs-real-cause misses are untuned

**Next**
- Capture real incidents to make the precision gate meaningful
- Confirmed-verdict feedback so the history prior learns from right answers only
- Down-weight incidental config tweaks (the 4 eval misses)
- Diagnosis only by design: no auto-rollback or auto-retraining in v1

---

# Thank You

**Skynet_DS: ML Reliability Copilot**

Diagnose, don't just detect.

Questions?
