# src/retrieval/evidence.py
from src.memory.store import Store
from src.observability.base import TelemetryAdapter
from src.schemas.telemetry import NormalizedRun


class EvidenceRetriever:
    """Query layer over observability/ + memory/ - 'get me run X', 'get me
    the last 5 runs for model Y', 'get me the baseline for model Y'.
    agents/ and tools/ should go through this, never the adapter/store
    directly (Architecture.md section 2, retrieval/)."""

    def __init__(self, adapter: TelemetryAdapter, store: Store) -> None:
        self._adapter = adapter
        self._store = store

    def get_run(self, run_id: str) -> NormalizedRun:
        return self._adapter.get_run(run_id)

    def get_recent_runs(self, model_id: str, limit: int = 5) -> list[NormalizedRun]:
        return self._adapter.get_recent_runs(model_id, limit)

    def get_baseline(self, model_id: str) -> NormalizedRun | None:
        cached = self._store.get_baseline(model_id)
        if cached is not None:
            return cached
        baseline = self._adapter.get_baseline(model_id)
        if baseline is not None and hasattr(self._store, "set_baseline"):
            self._store.set_baseline(model_id, baseline)
        return baseline

    def get_code_diff(self, base_ref: str | None, head_ref: str | None) -> str | None:
        """Phase 1 will route this through github/ for a real commit diff
        (BluePrint.md Build Order). Phase 0 stub keeps the call shape stable."""
        if not base_ref or not head_ref or base_ref == head_ref:
            return None
        return f"(stub) commit range {base_ref}..{head_ref} not fetched yet - github/ adapter is Phase 1"
