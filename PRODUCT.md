# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users
ML engineers who own models and the data/code pipelines behind them. They open the dashboard to triage: did the latest PR review pass, is a monitored model drifting, and which feature explains it. Confirmed by the user: the primary user is the triaging ML engineer; the dashboard must also be presentable to evaluators (secondary, not confirmed beyond that).

## Product Purpose
Skynet_DS reviews pull requests for ML/data-science-specific flaws (multi-agent: security, architecture, ML-debugging, drift) and diagnoses why a deployed model regressed or drifted (statistical drift engine + SHAP root-cause attribution). The dashboard (`demo-ui/`, NestJS + static vanilla HTML/CSS/JS) is a read-only presentation layer over what the Python core produces. Success: an engineer sees verdict, severity breakdown, drifted-feature share and top cause within one viewport and can drill into evidence.

## Positioning
One diagnostic core with two triggers (PR webhook and the continuous monitoring sweep), pairing detector-level drift evidence with ranked feature attribution - not generic linting.

## Operating Context
Data comes from the shared data directory: latest PR review (`data/prs/*/_review.json`), latest monitoring-sweep drift report and root cause (`data/monitoring/*.drift.json`, `*.root_cause.json`). Panels without live data show bundled sample data; the page must label which is which (X-Data-Source header, `#data-source` footer behavior).

## Capabilities and Constraints
- API contract is fixed: `GET /api/pr-review` (FinalPRReview), `/api/drift` (DatasetDriftReport incl. computed n_columns, n_drifted, share_drifted, dataset_drifted), `/api/root-cause` (DriftRootCause); `X-Data-Source: live|sample`.
- Frontend stays static vanilla HTML/CSS/JS served by NestJS; no frontend build step or framework (user choice: dark-first with a light theme toggle).
- First screen must surface: health verdict + KPIs, findings table, drift + attribution, live-vs-sample status (user-confirmed priorities).
- No login; read-only.

## Evidence on Hand
Sample data in `demo-ui/src/review/sample-data.ts`; real schema-shaped JSON from the Python core. `src/observability/exporter.py` provides `observe()` for a real model server to publish `mlcop_feature_value`; until one does, live drift panels are empty (a simulated demo exporter exists, synthetic data only).

## Product Principles
1. Verdict first, evidence on demand.
2. Never present sample data as live.
3. Mirror the Python schemas; the UI invents no fields.
4. Dense and scannable over decorative.

## Brand Commitments
Name: Skynet_DS. User-requested reference class: LLM-observability products (Langfuse, LangSmith, Helicone, Arize Phoenix).
