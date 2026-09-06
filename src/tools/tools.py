# src/tools/tools.py
"""LLM tool-call implementations exposed to agents/ (Architecture.md
section 2, tools/). Thin wrappers around retrieval/ - no business logic
here, just the surface an LLM agent calls by name."""

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
