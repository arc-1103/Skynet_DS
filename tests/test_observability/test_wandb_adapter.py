"""The W&B public API needs a network and an API key, so these drive the
adapter with stand-ins shaped like wandb.apis.public.Run/Api."""

from types import SimpleNamespace

import pytest
from wandb.errors import CommError

from src.observability.base import build_adapter
from src.observability.fake_adapter import FakeTelemetryAdapter
from src.observability.wandb_adapter import BASELINE_TAG, WandbTelemetryAdapter


def _run(run_id, created_at, lr, acc, commit="abc123"):
    return SimpleNamespace(
        id=run_id,
        entity="acme",
        project="fraud-clf",
        created_at=created_at,
        commit=commit,
        config={"learning_rate": lr, "data_version": "d3", "_wandb": {"cli_version": "x"}},
        summary_metrics={"acc": acc, "_runtime": 12.5, "_step": 9, "confusion": {"tp": 3}, "done": True},
    )


class _FakeApi:
    def __init__(self, runs):
        self._runs = {f"acme/fraud-clf/{r.id}": r for r in runs}
        self.runs_calls = []

    def run(self, path):
        if path not in self._runs:
            raise CommError(f"Could not find run {path}")
        return self._runs[path]

    def runs(self, path, filters=None, order="+created_at", per_page=50):
        self.runs_calls.append((path, filters, order, per_page))
        if path != "acme/fraud-clf":
            raise ValueError(f"Could not find project {path}")
        matches = list(self._runs.values())
        if filters:  # only the baseline query filters; tag membership is what matters here
            matches = [r for r in matches if r.id == "base"]
        return iter(sorted(matches, key=lambda r: r.created_at, reverse=True))


@pytest.fixture
def api():
    return _FakeApi(
        [
            _run("base", "2026-09-01T10:00:00", 3e-4, 0.91, commit="abc111"),
            _run("cur", "2026-09-02T10:00:00Z", 1e-3, 0.83),
        ]
    )


def test_get_run_normalizes_a_wandb_run(api):
    run = WandbTelemetryAdapter("acme", api=api).get_run("fraud-clf/cur")

    assert run.run_id == "acme/fraud-clf/cur"
    assert run.model_id == "fraud-clf"
    # internal "_" keys, non-numeric and boolean summaries are not metrics
    assert run.metrics == {"acc": 0.83}
    assert run.config == {"learning_rate": 1e-3}
    assert run.data_version == "d3"
    assert run.code_ref == "abc123"
    assert run.timestamp.tzinfo is not None


def test_get_run_accepts_the_full_path_it_hands_out(api):
    adapter = WandbTelemetryAdapter(api=api)  # no entity configured
    run = adapter.get_run("acme/fraud-clf/base")
    assert adapter.get_run(run.run_id).run_id == run.run_id


def test_get_run_unknown_id_raises_key_error(api):
    with pytest.raises(KeyError):
        WandbTelemetryAdapter("acme", api=api).get_run("fraud-clf/nope")


def test_get_recent_runs_newest_first_and_limited(api):
    runs = WandbTelemetryAdapter("acme", api=api).get_recent_runs("fraud-clf", limit=1)
    assert [r.run_id for r in runs] == ["acme/fraud-clf/cur"]
    assert api.runs_calls[-1] == ("acme/fraud-clf", None, "-created_at", 1)


def test_get_baseline_queries_the_tagged_finished_run(api):
    baseline = WandbTelemetryAdapter("acme", api=api).get_baseline("fraud-clf")

    assert baseline.run_id == "acme/fraud-clf/base"
    _, filters, _, _ = api.runs_calls[-1]
    assert filters == {"tags": {"$in": [BASELINE_TAG]}, "state": "finished"}


def test_unknown_project_has_no_runs_or_baseline(api):
    adapter = WandbTelemetryAdapter("acme", api=api)
    assert adapter.get_recent_runs("nope") == []
    assert adapter.get_baseline("nope") is None


def test_build_adapter_selects_backend(monkeypatch):
    from src.observability import wandb_adapter

    monkeypatch.setattr(wandb_adapter, "Api", lambda: _FakeApi([]))
    assert isinstance(build_adapter("fake"), FakeTelemetryAdapter)
    assert isinstance(build_adapter("wandb", "acme"), WandbTelemetryAdapter)
    with pytest.raises(ValueError, match="unknown TELEMETRY_BACKEND"):
        build_adapter("mlflow")
