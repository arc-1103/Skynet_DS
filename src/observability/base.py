# src/observability/base.py
from datetime import datetime
from typing import Protocol

from src.schemas.telemetry import FeatureSample, NormalizedRun


class TelemetryAdapter(Protocol):
    """Interface every run-telemetry backend adapter implements. Nothing in
    graph/, agents/, or tools/ should import wandb/httpx-to-Prometheus/etc.
    directly - if you find yourself doing that, this boundary has leaked
    (Architecture.md section 4)."""

    def get_run(self, run_id: str) -> NormalizedRun: ...

    def get_recent_runs(self, model_id: str, limit: int = 5) -> list[NormalizedRun]: ...

    def get_baseline(self, model_id: str) -> NormalizedRun | None: ...


class DistributionAdapter(Protocol):
    """Live feature distributions for drift detection (Workflows 2 and 5).
    Separate from TelemetryAdapter because a metrics store like Prometheus
    has time series, not training runs."""

    def get_feature_samples(self, model_id: str, start: datetime, end: datetime) -> FeatureSample: ...


def build_adapter(backend: str, wandb_entity: str | None = None) -> TelemetryAdapter:
    """The one place a backend name becomes an adapter, so entry points
    (cli.py, worker/jobs.py) never import a concrete adapter themselves.
    Imports are lazy: the fake backend must not require wandb installed."""
    if backend == "fake":
        from src.observability.fake_adapter import FakeTelemetryAdapter

        return FakeTelemetryAdapter()
    if backend == "wandb":
        from src.observability.wandb_adapter import WandbTelemetryAdapter

        return WandbTelemetryAdapter(wandb_entity)
    raise ValueError(f"unknown TELEMETRY_BACKEND {backend!r} - expected fake or wandb")
