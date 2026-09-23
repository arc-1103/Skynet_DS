from src.memory.sqlite_store import SQLiteStore
from src.observability.fake_adapter import FakeTelemetryAdapter
from src.retrieval.evidence import EvidenceRetriever
from src.tools import tools


def _retriever(tmp_path):
    return EvidenceRetriever(FakeTelemetryAdapter(), SQLiteStore(str(tmp_path / "m.db")))


def test_get_run_metrics_returns_normalized_run(tmp_path):
    run = tools.get_run_metrics(_retriever(tmp_path), "run-004")
    assert run.run_id == "run-004"


def test_get_baseline_returns_baseline_run(tmp_path):
    baseline = tools.get_baseline(_retriever(tmp_path), "fraud-clf")
    assert baseline is not None
    assert baseline.run_id == "run-baseline"


def test_get_code_diff_returns_none_when_refs_match(tmp_path):
    assert tools.get_code_diff(_retriever(tmp_path), "abc", "abc") is None


def test_get_code_diff_returns_a_range_label_when_refs_differ(tmp_path):
    diff = tools.get_code_diff(_retriever(tmp_path), "abc111", "abc123")
    assert diff is not None


def _run(config):
    from datetime import datetime, timezone
    from src.schemas.telemetry import NormalizedRun

    return NormalizedRun(run_id="r", model_id="m", timestamp=datetime.now(timezone.utc), config=config)


def test_get_config_diff_covers_changed_added_and_removed_keys():
    base = _run({"lr": 0.1, "dropout": 0.2, "same": 1})
    head = _run({"lr": 0.3, "same": 1, "warmup": 10})
    assert tools.get_config_diff(base, head) == {
        "lr": (0.1, 0.3),
        "dropout": (0.2, None),
        "warmup": (None, 10),
    }


def test_get_metric_changes_ignores_moves_within_tolerance():
    base = _run({})
    base.metrics.update({"acc": 0.910, "loss": 0.0, "only_base": 1.0})
    head = _run({})
    head.metrics.update({"acc": 0.915, "loss": 0.2, "only_head": 5.0})

    # acc moved 0.5% (inside 1%); a move from 0 always counts; unshared keys are skipped
    assert tools.get_metric_changes(base, head) == {"loss": (0.0, 0.2)}
    assert tools.get_metric_changes(base, head, rel_tol=0.001) == {"acc": (0.910, 0.915), "loss": (0.0, 0.2)}
