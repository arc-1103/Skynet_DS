from datetime import datetime, timezone

from src.graph.causal import CauseCandidate
from src.memory.sqlite_store import SQLiteStore
from src.schemas.telemetry import NormalizedRun


def _make_run(run_id="r1", model_id="m1") -> NormalizedRun:
    return NormalizedRun(
        run_id=run_id,
        model_id=model_id,
        metrics={"accuracy": 0.9},
        config={"lr": 1e-3},
        timestamp=datetime.now(timezone.utc),
    )


def test_get_baseline_missing_returns_none(tmp_path):
    store = SQLiteStore(str(tmp_path / "m.db"))
    assert store.get_baseline("m1") is None


def test_set_and_get_baseline_roundtrips(tmp_path):
    store = SQLiteStore(str(tmp_path / "m.db"))
    run = _make_run()
    store.set_baseline("m1", run)
    fetched = store.get_baseline("m1")
    assert fetched == run


def test_set_baseline_upserts_existing_model(tmp_path):
    store = SQLiteStore(str(tmp_path / "m.db"))
    store.set_baseline("m1", _make_run(run_id="r1"))
    store.set_baseline("m1", _make_run(run_id="r2"))
    assert store.get_baseline("m1").run_id == "r2"


def test_save_and_get_diagnosis_history_newest_first(tmp_path):
    store = SQLiteStore(str(tmp_path / "m.db"))
    older = CauseCandidate(kind="config", description="older", confidence=0.5)
    newer = CauseCandidate(kind="data", description="newer", confidence=0.7)
    store.save_diagnosis("m1", older)
    store.save_diagnosis("m1", newer)

    history = store.get_diagnosis_history("m1")

    assert [c.description for c in history] == ["newer", "older"]


def test_diagnosis_history_scoped_per_model(tmp_path):
    store = SQLiteStore(str(tmp_path / "m.db"))
    store.save_diagnosis("m1", CauseCandidate(kind="config", description="for-m1", confidence=0.5))
    store.save_diagnosis("m2", CauseCandidate(kind="config", description="for-m2", confidence=0.5))

    assert [c.description for c in store.get_diagnosis_history("m1")] == ["for-m1"]
    assert [c.description for c in store.get_diagnosis_history("m2")] == ["for-m2"]


def test_diagnosis_history_empty_for_unknown_model(tmp_path):
    store = SQLiteStore(str(tmp_path / "m.db"))
    assert store.get_diagnosis_history("no-such-model") == []


def test_schema_persists_across_reconnect(tmp_path):
    db_path = str(tmp_path / "m.db")
    store1 = SQLiteStore(db_path)
    store1.set_baseline("m1", _make_run())
    store1.close()

    store2 = SQLiteStore(db_path)
    assert store2.get_baseline("m1") is not None
