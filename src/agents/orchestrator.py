# src/agents/orchestrator.py
"""Diagnostic core: plan -> gather -> hypothesize -> rank -> explain.

BluePrint.md recommends agents/ be a simple explicit state machine (not
LangGraph) so it stays clearly separate from graph/ (the causal-candidate
graph). This module implements Workflow 1 (On-Demand Debug), Workflow 4
(deploy-scoped, via diagnose(deploy_sha=...)), Workflow 3 (compare()) and
the diagnosis half of Workflow 2 (diagnose_drift()). Workflow 5 is pure
scheduling in worker/ over these same entry points."""

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from src.agents.llm_client import call_llm_text
from src.core.config import settings
from src.drift.root_cause import explain_drift, to_cause_candidates
from src.drift.tables import frame_from_features
from src.graph.causal import CausalGraph, CauseCandidate, rank_correlates
from src.memory.store import Store
from src.retrieval.evidence import EvidenceRetriever
from src.schemas.drift import DatasetDriftReport
from src.schemas.telemetry import FeatureSample, NormalizedRun
from src.tools import tools

_EXPLAIN_PROMPT = Path(__file__).resolve().parents[1] / "prompts" / "explain.md"
_EXPLAIN_SYSTEM = (
    "You explain ML model regressions to ML engineers. Use only the evidence "
    "you are given; never invent causes, numbers or commits."
)


@dataclass
class DiagnosisResult:
    run: NormalizedRun
    baseline: NormalizedRun | None
    ranked_candidates: list[CauseCandidate]
    explanation: str


def _hypothesize_config(current: NormalizedRun, baseline: NormalizedRun) -> CauseCandidate | None:
    changed = tools.get_config_diff(baseline, current)
    if not changed:
        return None
    desc = "; ".join(f"`{k}` {old} -> {new}" for k, (old, new) in changed.items())
    return CauseCandidate(
        kind="config",
        description=f"Config change: {desc}",
        confidence=0.8 if len(changed) == 1 else 0.6,
        evidence_refs=[current.run_id, baseline.run_id],
    )


def _hypothesize_data(current: NormalizedRun, baseline: NormalizedRun) -> CauseCandidate:
    if current.data_version and baseline.data_version and current.data_version != baseline.data_version:
        return CauseCandidate(
            kind="data",
            description=f"Data version changed: {baseline.data_version} -> {current.data_version}",
            confidence=0.5,
            evidence_refs=[current.run_id, baseline.run_id],
        )
    return CauseCandidate(
        kind="data",
        description="No data version change detected",
        confidence=0.05,
        evidence_refs=[current.run_id, baseline.run_id],
    )


def _hypothesize_code(
    current: NormalizedRun,
    baseline: NormalizedRun,
    diff_summary: str | None,
    head_ref: str | None = None,
) -> CauseCandidate | None:
    """head_ref overrides current.code_ref - Workflow 4 scopes the code
    evidence to the deployed commit, not whatever the run was trained on."""
    head_ref = head_ref or current.code_ref
    if not (head_ref and baseline.code_ref and head_ref != baseline.code_ref):
        return None
    desc = f"Code changed: {baseline.code_ref} -> {head_ref}"
    if diff_summary:
        desc += f" ({diff_summary})"
    return CauseCandidate(
        kind="code",
        description=desc,
        confidence=0.3,
        evidence_refs=[current.run_id, baseline.run_id],
    )


def _explain(current: NormalizedRun, baseline: NormalizedRun | None, ranked: list[CauseCandidate]) -> str:
    if not baseline:
        return f"No baseline found for model '{current.model_id}' - cannot compute a delta."
    if not ranked:
        return f"Run '{current.run_id}' shows no meaningful delta vs. baseline '{baseline.run_id}'."

    metric_deltas = [
        f"{k} {baseline.metrics[k]:.3f} -> {v:.3f}"
        for k, v in current.metrics.items()
        if k in baseline.metrics
    ]
    metric_line = ", ".join(metric_deltas) if metric_deltas else "no shared metrics"

    lines = [f"Run '{current.run_id}' vs baseline '{baseline.run_id}': {metric_line}.", *_ranked_lines(ranked)]
    return "\n".join(lines)


def _ranked_lines(ranked: list[CauseCandidate]) -> list[str]:
    top = ranked[0]
    lines = [f"Most likely cause ({top.kind}, confidence {top.confidence:.2f}): {top.description}"]
    for candidate in ranked[1:]:
        if candidate.confidence > 0.05:
            lines.append(
                f"Secondary candidate ({candidate.kind}, confidence {candidate.confidence:.2f}): {candidate.description}"
            )
    return lines


def _llm_explain(current: NormalizedRun, baseline: NormalizedRun | None, ranked: list[CauseCandidate]) -> str | None:
    """Workflow 1 step 6: prompts/explain.md through the LLM. None -> the
    caller uses the deterministic _explain() template (no key configured,
    call failed, or the answer fails the sanity check below)."""
    # opt-in: this sends run configs, metrics and commit SHAs to the LLM
    # provider, which GROQ_API_KEY alone (set for the PR agents) never agreed to
    if not (settings.llm_explain and baseline and ranked):
        return None  # the template's fixed no-baseline/no-delta lines are already exact
    prompt = _EXPLAIN_PROMPT.read_text(encoding="utf-8")
    # Plain replace, not str.format: prompts/ is meant to be edited freely
    # (Architecture.md), and one literal "{" in a future prompt would make
    # format() raise inside every diagnose().
    for placeholder, value in {
        "{current_run}": current.model_dump_json(),
        "{baseline_run}": baseline.model_dump_json(),
        "{ranked_candidates}": json.dumps([c.model_dump() for c in ranked]),
    }.items():
        prompt = prompt.replace(placeholder, value)
    text = call_llm_text(_EXPLAIN_SYSTEM, prompt)
    # cheap guard against an answer about something else entirely
    return text if text and current.run_id in text else None


def diagnose(
    run_id: str,
    retriever: EvidenceRetriever,
    store: Store,
    deploy_sha: str | None = None,
) -> DiagnosisResult:
    """Workflow 1 (On-Demand Debug): mlcop debug --run-id <id>.

    With deploy_sha it is Workflow 4 (Deployment Regression Triage): the
    code candidate is scoped to baseline.code_ref..deploy_sha, the exact
    commit range the deploy shipped."""
    current = tools.get_run_metrics(retriever, run_id)
    baseline = tools.get_baseline(retriever, current.model_id)

    graph = CausalGraph()
    # No metric moved -> nothing to explain. Without this, any config/code
    # difference was reported as "most likely cause" of a regression that
    # never happened (e.g. every deploy with no new training run).
    if baseline and tools.get_metric_changes(baseline, current):
        head_ref = deploy_sha or current.code_ref
        diff_summary = tools.get_code_diff(retriever, baseline.code_ref, head_ref)
        for candidate in (
            _hypothesize_config(current, baseline),
            _hypothesize_data(current, baseline),
            _hypothesize_code(current, baseline, diff_summary, head_ref),
        ):
            if candidate:
                graph.add_candidate(candidate)

    # history is read before this verdict is saved, so a diagnosis never
    # counts as evidence for itself
    ranked = graph.rank(history=tools.get_diagnosis_history(retriever, current.model_id))
    explanation = _llm_explain(current, baseline, ranked) or _explain(current, baseline, ranked)

    if ranked:
        store.save_diagnosis(current.model_id, ranked[0])

    return DiagnosisResult(run=current, baseline=baseline, ranked_candidates=ranked, explanation=explanation)


@dataclass
class ComparisonResult:
    runs: list[NormalizedRun]
    metric: str
    ranked_candidates: list[CauseCandidate]
    explanation: str


def compare(
    run_ids: list[str],
    retriever: EvidenceRetriever,
    store: Store,
    metric: str | None = None,
) -> ComparisonResult:
    """Workflow 3 (Experiment Comparison): mlcop compare --runs A,B,C.
    Ranks which changed variable best explains the metric spread across
    all N runs (graph.rank_correlates), not one baseline vs. one current."""
    if len(run_ids) < 2:
        raise ValueError("compare needs at least 2 run ids")
    runs = [tools.get_run_metrics(retriever, run_id) for run_id in run_ids]
    metric = metric or next(iter(runs[0].metrics), None)
    if metric is None or not all(metric in r.metrics for r in runs):
        raise ValueError(f"metric {metric!r} is not logged on every run")

    ranked = rank_correlates(runs, metric)
    values = ", ".join(f"{r.run_id}={r.metrics[metric]:.3f}" for r in runs)
    lines = [f"{metric} across {len(runs)} runs: {values}."]
    if ranked:
        top = ranked[0]
        lines.append(f"Strongest correlate ({top.kind}, confidence {top.confidence:.2f}): {top.description}")
        lines += [
            f"Also varies ({c.kind}, confidence {c.confidence:.2f}): {c.description}"
            for c in ranked[1:3]
        ]
        # one comparison can span models; file the verdict under each
        for model_id in dict.fromkeys(r.model_id for r in runs):
            store.save_diagnosis(model_id, top)
    else:
        lines.append("No variable changes across these runs that explains a metric difference.")

    return ComparisonResult(runs=runs, metric=metric, ranked_candidates=ranked, explanation="\n".join(lines))


@dataclass
class DriftDiagnosis:
    model_id: str
    report: DatasetDriftReport
    ranked_candidates: list[CauseCandidate]
    explanation: str


# A drift of the same feature set seen in this many earlier episodes reads
# as expected variation rather than an incident (Workflow 2 step 5).
_SEASONAL_AFTER = 2
# How far before a drift's window a pipeline commit still counts as a
# suspect. Bounded because "since the baseline" grows forever and ends up
# blaming every drift on whatever the pipeline last touched.
# ponytail: one day covers same-day merge->deploy lag; widen if pipeline
# changes routinely take longer to reach production data.
PIPELINE_LOOKBACK = timedelta(days=1)


def drift_signature(report: DatasetDriftReport) -> str:
    """Identity of a drift: which features moved. worker/ uses it to tell an
    ongoing episode from a new one; history uses it to spot recurrences."""
    return "drift-features:" + ",".join(sorted(c.column for c in report.columns if c.drifted))


def diagnose_drift(
    baseline: FeatureSample,
    current: FeatureSample,
    report: DatasetDriftReport,
    retriever: EvidenceRetriever,
    store: Store,
    pipeline_paths: tuple[str, ...] = (),
) -> DriftDiagnosis:
    """Workflow 2 steps 4-6, called by worker/ once per drift *episode*
    (not per check - see worker/jobs.check_drift). Ranks three explanations
    against each other: a data-pipeline code change, a genuine upstream data
    shift, and recurring (seasonal) variation this model has shown before."""
    model_id = current.model_id
    drifted = sorted(c.column for c in report.columns if c.drifted)
    # every candidate carries the signature, so whichever one wins is
    # findable in history as "this same drift happened in an earlier episode"
    signature = drift_signature(report)
    refs = [signature, f"window:{current.start.isoformat()}..{current.end.isoformat()}"]

    graph = CausalGraph()

    detail = ""
    try:
        attribution = explain_drift(
            frame_from_features(baseline.features), frame_from_features(current.features), label=model_id
        )
        drivers = to_cause_candidates(attribution, top_n=3)
        if drivers:
            detail = "; drivers: " + " | ".join(d.description for d in drivers)
    except ValueError:  # too few rows to fit the SHAP surrogate - keep the detector verdict alone
        pass
    graph.add_candidate(
        CauseCandidate(
            kind="data",
            description=f"Upstream data shift in {', '.join(drifted)} ({report.share_drifted:.0%} of features){detail}",
            confidence=round(0.3 + 0.4 * report.share_drifted, 3),
            evidence_refs=refs,
        )
    )

    since = max(baseline.end, current.start - PIPELINE_LOOKBACK)
    commits = tools.get_pipeline_commits(retriever, pipeline_paths, since)
    if commits:
        graph.add_candidate(
            CauseCandidate(
                kind="code",
                description=f"Data pipeline changed shortly before the drift: {'; '.join(commits[:3])}",
                confidence=0.7,
                evidence_refs=refs,
            )
        )

    history = tools.get_diagnosis_history(retriever, model_id)
    # One history entry per episode, so this counts past episodes, not sweeps.
    # ponytail: counts recurrences of the same drifted-feature set, not true
    # periodicity; add timestamps to the check if weekly/monthly cycles matter
    recurrences = sum(signature in past.evidence_refs for past in history)
    if recurrences >= _SEASONAL_AFTER:
        graph.add_candidate(
            CauseCandidate(
                kind="data",
                description=f"Expected/seasonal variation: the same features drifted in {recurrences} earlier episodes",
                confidence=min(0.9, 0.3 + 0.2 * recurrences),
                evidence_refs=refs,
            )
        )

    ranked = graph.rank(history=history)
    header = (
        f"Drift on '{model_id}': {report.n_drifted}/{report.n_columns} features drifted "
        f"({report.share_drifted:.0%}): {', '.join(drifted)}."
    )
    if commits is None:
        header += " (Pipeline commits not checked - set GITHUB_REPO and MLCOP_PIPELINE_PATHS.)"
    store.save_diagnosis(model_id, ranked[0])
    return DriftDiagnosis(
        model_id=model_id,
        report=report,
        ranked_candidates=ranked,
        explanation="\n".join([header, *_ranked_lines(ranked)]),
    )
