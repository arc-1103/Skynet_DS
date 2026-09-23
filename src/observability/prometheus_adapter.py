# src/observability/prometheus_adapter.py
"""Prometheus distribution adapter - live feature values for drift
detection and continuous monitoring (Workflows 2 and 5, Architecture.md
section 4).

Prometheus holds time series, not training runs, so this adapter does not
implement TelemetryAdapter (runs come from W&B); it implements
DistributionAdapter and normalizes into schemas/telemetry.FeatureSample.

Export convention the model server must follow - one gauge per feature:
    mlcop_feature_value{model="<model_id>", feature="<feature name>"}
A range query over the window turns each series into that feature's
sample of observed values.

This is the only module that talks to Prometheus."""

import math
from datetime import datetime

import httpx

from src.schemas.telemetry import FeatureSample

DEFAULT_METRIC = "mlcop_feature_value"
# Prometheus refuses range queries over 11,000 points per series.
_TARGET_POINTS = 1000
# ponytail: assumes the common 15s scrape interval; a finer step only
# returns duplicate points, a coarser scrape just yields fewer samples.
_MIN_STEP_SECONDS = 15


def _label_value(value: str) -> str:
    """Escape a value for a PromQL double-quoted label matcher."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


class PrometheusAdapter:
    def __init__(
        self,
        base_url: str,
        metric: str = DEFAULT_METRIC,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._metric = metric
        # A client is injectable for tests (httpx.MockTransport); without one
        # each query uses httpx's one-shot get(), which closes its own
        # connection - an adapter built per drift check must not leak a pool.
        self._client = client

    def get_feature_samples(self, model_id: str, start: datetime, end: datetime) -> FeatureSample:
        span = (end - start).total_seconds()
        response = (self._client or httpx).get(
            f"{self._base_url}/api/v1/query_range",
            params={
                "query": f'{self._metric}{{model="{_label_value(model_id)}"}}',
                "start": start.timestamp(),
                "end": end.timestamp(),
                "step": max(_MIN_STEP_SECONDS, int(span // _TARGET_POINTS)),
            },
            timeout=10.0,
        )
        response.raise_for_status()
        body = response.json()
        if body.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {body.get('error', body)}")

        features: dict[str, list[float]] = {}
        for series in body["data"]["result"]:
            name = series["metric"].get("feature")
            if not name:
                continue
            values = (float(v) for _, v in series["values"])
            features.setdefault(name, []).extend(v for v in values if math.isfinite(v))
        return FeatureSample(model_id=model_id, start=start, end=end, features=features)
