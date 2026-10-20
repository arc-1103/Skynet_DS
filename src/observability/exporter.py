# src/observability/exporter.py
"""Producer side of the mlcop_feature_value contract that
prometheus_adapter.py reads: one gauge per (model, feature), set to the
value of that feature on the latest prediction.

Call observe() from the model server's predict path, once per request:

    from src.observability.exporter import observe, serve
    serve(9090)                                  # once, at startup
    observe("fraud-clf", {"amount": 42.0, "age": 31})

Prometheus scrapes the gauge every scrape_interval, so the drift window is
a sample of requests (one per scrape), not every request - plenty to
compare distributions, not an audit log.

`python -m src.observability.exporter` runs a *simulated* model instead
(demo only): it emits synthetic features that drift after --drift-after
seconds, so the drift panels have something to show without a real model.
Never point a real dashboard at it and call the result live."""

import argparse
import math
import random
import time
from collections.abc import Mapping

from prometheus_client import Gauge, start_http_server

from src.observability.prometheus_adapter import DEFAULT_METRIC

FEATURE_VALUE = Gauge(DEFAULT_METRIC, "Latest observed value of a model input feature", ["model", "feature"])


def observe(model_id: str, features: Mapping[str, float]) -> None:
    """Record one prediction's numeric features. Non-finite values are
    skipped: a NaN would poison the scraped series, and the adapter drops
    them on read anyway."""
    for name, value in features.items():
        value = float(value)
        if math.isfinite(value):
            FEATURE_VALUE.labels(model=model_id, feature=name).set(value)


def serve(port: int) -> None:
    """Expose /metrics on `port` in a background thread."""
    start_http_server(port)


def simulated_features(elapsed: float, drift_after: float, rng: random.Random) -> dict[str, float]:
    """`amount` shifts (mean 50 -> 80) once `drift_after` seconds have
    passed; `age` stays put, so there is one real culprit to attribute."""
    drifted = elapsed >= drift_after
    return {
        "amount": rng.gauss(80 if drifted else 50, 10),
        "age": rng.gauss(35, 8),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulated model exporting mlcop_feature_value (demo only)")
    parser.add_argument("--model", default="fraud-clf")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--drift-after", type=float, default=600, help="seconds until `amount` drifts")
    args = parser.parse_args()

    serve(args.port)
    rng, started = random.Random(0), time.monotonic()
    print(f"simulating {args.model} on :{args.port}/metrics, drift after {args.drift_after:.0f}s")
    while True:
        observe(args.model, simulated_features(time.monotonic() - started, args.drift_after, rng))
        time.sleep(1)


if __name__ == "__main__":
    main()
