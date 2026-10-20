import random

import httpx
from prometheus_client import generate_latest

from src.observability.exporter import FEATURE_VALUE, observe, simulated_features
from src.observability.prometheus_adapter import PrometheusAdapter


def test_observe_sets_one_gauge_per_feature():
    observe("m-obs", {"amount": 42.0, "age": 31})
    assert FEATURE_VALUE.labels(model="m-obs", feature="amount")._value.get() == 42.0
    assert FEATURE_VALUE.labels(model="m-obs", feature="age")._value.get() == 31.0


def test_observe_skips_non_finite_values():
    observe("m-nan", {"ok": 1.0, "bad": float("nan"), "inf": float("inf")})
    lines = [l for l in generate_latest().decode().splitlines() if 'model="m-nan"' in l]
    assert len(lines) == 1 and 'feature="ok"' in lines[0]


def test_exposed_series_is_what_the_adapter_queries_for():
    """The adapter reads `mlcop_feature_value{model=...}` and takes the
    `feature` label as the column name - the exporter must emit that shape."""
    observe("m-contract", {"amount": 7.5})
    line = next(
        l for l in generate_latest().decode().splitlines() if l.startswith("mlcop_feature_value{") and "m-contract" in l
    )
    assert 'feature="amount"' in line and line.endswith(" 7.5")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["query"] == 'mlcop_feature_value{model="m-contract"}'
        series = {"metric": {"feature": "amount"}, "values": [[1, "7.5"], [2, "8.5"]]}
        return httpx.Response(200, json={"status": "success", "data": {"result": [series]}})

    from datetime import datetime, timezone

    adapter = PrometheusAdapter("http://prom", client=httpx.Client(transport=httpx.MockTransport(handler)))
    now = datetime.now(timezone.utc)
    assert adapter.get_feature_samples("m-contract", now, now).features == {"amount": [7.5, 8.5]}


def test_simulated_amount_drifts_but_age_does_not():
    def mean(feature, elapsed):
        rng = random.Random(1)
        return sum(simulated_features(elapsed, 100, rng)[feature] for _ in range(500)) / 500

    assert mean("amount", 0) < 60 < mean("amount", 200)
    assert abs(mean("age", 0) - mean("age", 200)) < 2
