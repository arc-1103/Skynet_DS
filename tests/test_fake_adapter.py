import pytest

from src.observability.fake_adapter import FakeTelemetryAdapter


def test_get_run_returns_expected_run():
    adapter = FakeTelemetryAdapter()
    run = adapter.get_run("run-004")
    assert run.run_id == "run-004"
    assert run.model_id == "fraud-clf"


def test_get_run_unknown_id_raises_key_error():
    adapter = FakeTelemetryAdapter()
    with pytest.raises(KeyError):
        adapter.get_run("nope")


def test_get_recent_runs_filters_by_model_and_sorts_newest_first():
    adapter = FakeTelemetryAdapter()
    runs = adapter.get_recent_runs("fraud-clf", limit=5)
    assert [r.run_id for r in runs] == ["run-004", "run-baseline"]


def test_get_recent_runs_respects_limit():
    adapter = FakeTelemetryAdapter()
    runs = adapter.get_recent_runs("fraud-clf", limit=1)
    assert len(runs) == 1
    assert runs[0].run_id == "run-004"


def test_get_recent_runs_unknown_model_returns_empty():
    adapter = FakeTelemetryAdapter()
    assert adapter.get_recent_runs("no-such-model") == []


def test_get_baseline_known_model():
    adapter = FakeTelemetryAdapter()
    baseline = adapter.get_baseline("fraud-clf")
    assert baseline is not None
    assert baseline.run_id == "run-baseline"


def test_get_baseline_unknown_model_returns_none():
    adapter = FakeTelemetryAdapter()
    assert adapter.get_baseline("no-such-model") is None
