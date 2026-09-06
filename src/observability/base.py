# src/observability/base.py
from typing import Protocol
from src.schemas.telemetry import NormalizedRun


class TelemetryAdapter(Protocol):
    """Interface every telemetry backend adapter implements. Nothing in graph/,
    agents/, or tools/ should import mlflow/prometheus_client/etc. directly -
    if you find yourself doing that, this boundary has leaked
    (Architecture.md section 4)."""

    def get_run(self, run_id: str) -> NormalizedRun: ...

    def get_recent_runs(self, model_id: str, limit: int = 5) -> list[NormalizedRun]: ...

    def get_baseline(self, model_id: str) -> NormalizedRun | None: ...
