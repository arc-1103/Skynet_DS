# src/tools/tools.py
"""LLM tool-call implementations exposed to agents/ (Architecture.md
section 2, tools/). Thin wrappers around retrieval/ - no business logic
here, just the surface an LLM agent calls by name."""

from datetime import datetime
from typing import Any

from src.graph.causal import CauseCandidate
from src.retrieval.evidence import EvidenceRetriever
from src.schemas.telemetry import NormalizedRun


def get_run_metrics(retriever: EvidenceRetriever, run_id: str) -> NormalizedRun:
    """Fetch metrics + config for a single run."""
    return retriever.get_run(run_id)


def get_baseline(retriever: EvidenceRetriever, model_id: str) -> NormalizedRun | None:
    """Fetch the last known-good run for a model."""
    return retriever.get_baseline(model_id)


def get_code_diff(retriever: EvidenceRetriever, base_ref: str | None, head_ref: str | None) -> str | None:
    """Fetch the commit diff between two code refs, if they differ."""
    return retriever.get_code_diff(base_ref, head_ref)


def get_config_diff(base: NormalizedRun, head: NormalizedRun) -> dict[str, tuple[Any, Any]]:
    """Config keys that differ between two runs -> (base value, head value).
    Covers keys added or removed, not just changed (None on the missing side)."""
    keys = dict.fromkeys([*base.config, *head.config])
    return {
        k: (base.config.get(k), head.config.get(k))
        for k in keys
        if base.config.get(k) != head.config.get(k)
    }


def get_diagnosis_history(retriever: EvidenceRetriever, model_id: str) -> list[CauseCandidate]:
    """Past top verdicts for a model, newest first."""
    return retriever.get_diagnosis_history(model_id)


def get_pipeline_commits(retriever: EvidenceRetriever, paths: tuple[str, ...], since: datetime) -> list[str] | None:
    """Recent commits to the data pipeline paths (None = could not check)."""
    return retriever.get_pipeline_commits(paths, since)


def get_metric_changes(
    base: NormalizedRun, head: NormalizedRun, rel_tol: float = 0.01
) -> dict[str, tuple[float, float]]:
    """Shared metrics that moved by more than rel_tol (relative to the base
    value) -> (base value, head value). Empty means there is no delta to
    explain. Direction-agnostic: which way is "worse" depends on the metric."""
    return {
        k: (base.metrics[k], v)
        for k, v in head.metrics.items()
        if k in base.metrics and abs(v - base.metrics[k]) > rel_tol * abs(base.metrics[k])
    }
