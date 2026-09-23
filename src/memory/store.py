# src/memory/store.py
from typing import Protocol

from src.graph.causal import CauseCandidate
from src.schemas.telemetry import FeatureSample, NormalizedRun


class Store(Protocol):
    """Storage-agnostic interface for baselines and diagnosis history. v1
    implementation is SQLite/local disk; this is the seam pulled on for
    hosted/multi-tenant later (Architecture.md section 2, memory/)."""

    def get_baseline(self, model_id: str) -> NormalizedRun | None: ...

    def save_diagnosis(self, model_id: str, verdict: CauseCandidate) -> None: ...

    def get_diagnosis_history(self, model_id: str) -> list[CauseCandidate]: ...

    def get_baseline_distribution(self, model_id: str) -> FeatureSample | None: ...

    def set_baseline_distribution(self, model_id: str, sample: FeatureSample) -> None: ...

    def get_active_drift(self, model_id: str) -> str | None:
        """Signature of the drift episode currently open for a model, if any."""
        ...

    def set_active_drift(self, model_id: str, signature: str | None) -> None:
        """Open (signature) or close (None) a model's drift episode."""
        ...
