from datetime import datetime, timezone

import pytest

from src.memory.sqlite_store import SQLiteStore
from src.observability.fake_adapter import FakeTelemetryAdapter
from src.retrieval import evidence
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


def test_get_code_diff_without_a_repo_labels_the_range(tmp_path):
    retriever = EvidenceRetriever(FakeTelemetryAdapter(), SQLiteStore(str(tmp_path / "m.db")))
    diff = retriever.get_code_diff("abc111", "abc123")
    assert "abc111..abc123" in diff
    assert "no GITHUB_REPO" in diff


def test_get_code_diff_fetches_from_github_when_a_repo_is_configured(tmp_path, monkeypatch):
    calls = []

    def fake_fetch(repo, base, head, token):
        calls.append((repo, base, head, token))
        return "1 commit(s), 1 file(s) changed (+3/-1): train.py"

    monkeypatch.setattr(evidence, "fetch_commit_diff", fake_fetch)
    retriever = EvidenceRetriever(
        FakeTelemetryAdapter(), SQLiteStore(str(tmp_path / "m.db")), "acme/models", "tok"
    )

    diff = retriever.get_code_diff("abc111", "abc123")

    assert calls == [("acme/models", "abc111", "abc123", "tok")]
    assert diff.endswith("train.py")


def test_get_code_diff_fetch_failure_does_not_raise(tmp_path, monkeypatch):
    def boom(*a):
        raise ConnectionError("github down")

    monkeypatch.setattr(evidence, "fetch_commit_diff", boom)
    retriever = EvidenceRetriever(FakeTelemetryAdapter(), SQLiteStore(str(tmp_path / "m.db")), "acme/models")

    assert retriever.get_code_diff("abc111", "abc123") == "commit range abc111..abc123, diff fetch failed"


def test_get_pipeline_commits_distinguishes_unchecked_from_none_found(tmp_path, monkeypatch):
    since = datetime.now(timezone.utc)
    store = SQLiteStore(str(tmp_path / "m.db"))
    monkeypatch.setattr(evidence, "fetch_path_commits", lambda repo, paths, since, token: [])

    # no repo or no paths configured -> could not check
    assert EvidenceRetriever(FakeTelemetryAdapter(), store).get_pipeline_commits(("etl/",), since) is None
    assert EvidenceRetriever(FakeTelemetryAdapter(), store, "acme/m").get_pipeline_commits((), since) is None
    # checked, nothing changed
    assert EvidenceRetriever(FakeTelemetryAdapter(), store, "acme/m").get_pipeline_commits(("etl/",), since) == []

    def boom(*a):
        raise ConnectionError("github down")

    monkeypatch.setattr(evidence, "fetch_path_commits", boom)
    assert EvidenceRetriever(FakeTelemetryAdapter(), store, "acme/m").get_pipeline_commits(("etl/",), since) is None


def test_get_code_diff_refuses_refs_that_are_not_shas(tmp_path, monkeypatch):
    """code_ref comes from run metadata anyone logging a run controls; it
    must not reach the GitHub compare URL unless it's a real SHA."""
    monkeypatch.setattr(evidence, "fetch_commit_diff", lambda *a: pytest.fail("called GitHub"))
    retriever = EvidenceRetriever(FakeTelemetryAdapter(), SQLiteStore(str(tmp_path / "m.db")), "acme/models")

    assert retriever.get_code_diff("abc111", "../../../orgs/x") == (
        "commit range abc111..../../../orgs/x, diff not fetched - refs are not commit SHAs"
    )
    assert "not commit SHAs" in retriever.get_code_diff("main", "abc123")
