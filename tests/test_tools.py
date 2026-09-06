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


def test_get_code_diff_returns_stub_when_refs_differ(tmp_path):
    diff = tools.get_code_diff(_retriever(tmp_path), "abc111", "abc123")
    assert diff is not None
