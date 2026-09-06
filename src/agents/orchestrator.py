# src/agents/orchestrator.py
"""Diagnostic core: plan -> gather -> hypothesize -> rank -> explain.

BluePrint.md recommends agents/ be a simple explicit state machine (not
LangGraph) so it stays clearly separate from graph/ (the causal-candidate
graph). This module implements Workflow 1 (On-Demand Debug) end-to-end;
Workflows 2-5 reuse this same diagnose() call with different triggers
(Workflow.md "Cross-Workflow Notes")."""

from dataclasses import dataclass

from src.graph.causal import CausalGraph, CauseCandidate
from src.memory.store import Store
from src.retrieval.evidence import EvidenceRetriever
from src.schemas.telemetry import NormalizedRun
from src.tools import tools


@dataclass
class DiagnosisResult:
    run: NormalizedRun
    baseline: NormalizedRun | None
    ranked_candidates: list[CauseCandidate]
    explanation: str


def _hypothesize_config(current: NormalizedRun, baseline: NormalizedRun) -> CauseCandidate | None:
    changed = {
        k: (baseline.config.get(k), v)
        for k, v in current.config.items()
        if baseline.config.get(k) != v
    }
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


def _hypothesize_code(current: NormalizedRun, baseline: NormalizedRun, diff_summary: str | None) -> CauseCandidate | None:
    if not (current.code_ref and baseline.code_ref and current.code_ref != baseline.code_ref):
        return None
    desc = f"Code changed: {baseline.code_ref} -> {current.code_ref}"
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

    top = ranked[0]
    lines = [
        f"Run '{current.run_id}' vs baseline '{baseline.run_id}': {metric_line}.",
        f"Most likely cause ({top.kind}, confidence {top.confidence:.2f}): {top.description}",
    ]
    for candidate in ranked[1:]:
        if candidate.confidence > 0.05:
            lines.append(
                f"Secondary candidate ({candidate.kind}, confidence {candidate.confidence:.2f}): {candidate.description}"
            )
    return "\n".join(lines)


def diagnose(run_id: str, retriever: EvidenceRetriever, store: Store) -> DiagnosisResult:
    """Workflow 1 (On-Demand Debug): mlcop debug --run-id <id>."""
    # plan: Phase 0's plan is fixed (single-run-vs-baseline). Phase 1+ branches
    # here for multi-run comparison (Workflow 3) and deploy-scoped ranges (Workflow 4).
    current = tools.get_run_metrics(retriever, run_id)
    baseline = tools.get_baseline(retriever, current.model_id)

    graph = CausalGraph()
    if baseline:
        diff_summary = tools.get_code_diff(retriever, baseline.code_ref, current.code_ref)
        for candidate in (
            _hypothesize_config(current, baseline),
            _hypothesize_data(current, baseline),
            _hypothesize_code(current, baseline, diff_summary),
        ):
            if candidate:
                graph.add_candidate(candidate)

    ranked = graph.rank()
    explanation = _explain(current, baseline, ranked)

    if ranked:
        store.save_diagnosis(current.model_id, ranked[0])

    return DiagnosisResult(run=current, baseline=baseline, ranked_candidates=ranked, explanation=explanation)
