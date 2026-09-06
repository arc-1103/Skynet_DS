from datetime import datetime, timezone

from src.memory.sqlite_store import SQLiteStore
from src.observability.fake_adapter import FakeTelemetryAdapter
from src.retrieval.evidence import EvidenceRetriever
from src.schemas.telemetry import NormalizedRun


def test_get_run_delegates_to_adapter(tmp_path):
    retriever = EvidenceRetriever(FakeTelemetryAdapter(), SQLiteStore(str(tmp_path / "m.db")))
    run = retriever.get_run("run-004")
    assert run.run_id == "run-004"


def test_get_recent_runs_delegates_to_adapter(tmp_path):
    retriever = EvidenceRetriever(FakeTelemetryAdapter(), SQLiteStore(str(tmp_path / "m.db")))
    runs = retriever.get_recent_runs("fraud-clf", limit=1)
    assert len(runs) == 1


def test_get_baseline_falls_back_to_adapter_and_caches_in_store(tmp_path):
    store = SQLiteStore(str(tmp_path / "m.db"))
    retriever = EvidenceRetriever(FakeTelemetryAdapter(), store)

    assert store.get_baseline("fraud-clf") is None  # not cached yet

    baseline = retriever.get_baseline("fraud-clf")

    assert baseline is not None
    assert baseline.run_id == "run-baseline"
    assert store.get_baseline("fraud-clf") == baseline  # now cached


def test_get_baseline_prefers_store_cache_over_adapter(tmp_path):
    store = SQLiteStore(str(tmp_path / "m.db"))
    cached = NormalizedRun(
        run_id="cached-run",
        model_id="fraud-clf",
        timestamp=datetime.now(timezone.utc),
    )
    store.set_baseline("fraud-clf", cached)
    retriever = EvidenceRetriever(FakeTelemetryAdapter(), store)

    baseline = retriever.get_baseline("fraud-clf")

    assert baseline.run_id == "cached-run"


def test_get_baseline_unknown_model_returns_none(tmp_path):
    retriever = EvidenceRetriever(FakeTelemetryAdapter(), SQLiteStore(str(tmp_path / "m.db")))
    assert retriever.get_baseline("no-such-model") is None


def test_get_code_diff_same_ref_returns_none(tmp_path):
    retriever = EvidenceRetriever(FakeTelemetryAdapter(), SQLiteStore(str(tmp_path / "m.db")))
    assert retriever.get_code_diff("abc", "abc") is None


def test_get_code_diff_missing_ref_returns_none(tmp_path):
    retriever = EvidenceRetriever(FakeTelemetryAdapter(), SQLiteStore(str(tmp_path / "m.db")))
    assert retriever.get_code_diff(None, "abc") is None
    assert retriever.get_code_diff("abc", None) is None


def test_get_code_diff_different_refs_returns_stub_message(tmp_path):
    retriever = EvidenceRetriever(FakeTelemetryAdapter(), SQLiteStore(str(tmp_path / "m.db")))
    diff = retriever.get_code_diff("abc111", "abc123")
    assert diff is not None
    assert "abc111..abc123" in diff
