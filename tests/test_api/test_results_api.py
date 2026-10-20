import json
import os

import pytest
from fastapi.testclient import TestClient

from src.api import webhooks
from src.core.config import Settings, get_settings
from src.schemas.drift import ColumnDriftReport, ColumnKind, DatasetDriftReport


@pytest.fixture
def results(tmp_path):
    """Client over a temp data dir; `use(**overrides)` swaps the settings."""
    prs, monitoring = tmp_path / "prs", tmp_path / "monitoring"

    def use(**overrides):
        settings = Settings(pr_cache_dir=str(prs), monitoring_dir=str(monitoring), **overrides)
        webhooks.app.dependency_overrides[get_settings] = lambda: settings

    use()
    client = TestClient(webhooks.app)
    client.prs, client.monitoring, client.use = prs, monitoring, use
    yield client
    webhooks.app.dependency_overrides.clear()


def _drift(label, drifted):
    column = ColumnDriftReport(column="amount", kind=ColumnKind.NUMERICAL, n_reference=10, n_current=10, drifted=drifted, vote="majority")
    return DatasetDriftReport(columns=[column], share_threshold=0.5, label=label).model_dump_json()


def test_nothing_written_yet_is_a_404(results):
    for path in ("/results/pr-review", "/results/drift", "/results/root-cause"):
        assert results.get(path).status_code == 404


def test_pr_review_joins_in_the_cached_pr_identity(results):
    repo = results.prs / "acme_fraud"
    repo.mkdir(parents=True)
    (repo / "pr_7_abc_review.json").write_text(json.dumps(
        {"labels": [], "overall_passed": True, "markdown_summary": "ok", "inline_findings": []}))
    (repo / "pr_7_abc.json").write_text(json.dumps(
        {"pr_id": 7, "title": "Tune lr", "repo_full_name": "acme/fraud", "commit_sha": "abc", "base_ref": "main"}))

    res = results.get("/results/pr-review")

    assert res.status_code == 200
    assert res.headers["x-data-updated"]
    body = res.json()
    assert (body["pr_id"], body["pr_title"], body["repo_full_name"]) == (7, "Tune lr", "acme/fraud")
    assert body["overall_passed"] is True


def test_newest_drift_file_wins_and_keeps_its_computed_verdict(results):
    results.monitoring.mkdir()
    old, new = results.monitoring / "a.drift.json", results.monitoring / "b.drift.json"
    old.write_text(_drift("old", drifted=False))
    new.write_text(_drift("new", drifted=True))
    os.utime(old, (1_000, 1_000))

    body = results.get("/results/drift").json()

    assert body["label"] == "new"
    assert body["dataset_drifted"] is True


def test_a_file_that_does_not_parse_is_a_503_not_a_fallback(results):
    results.monitoring.mkdir()
    (results.monitoring / "m.drift.json").write_text('{"columns": [')  # truncated mid-write

    res = results.get("/results/drift")

    assert res.status_code == 503
    assert "m.drift.json" in res.json()["detail"]


def test_token_is_required_once_configured(results):
    results.use(results_api_token="s3cret")

    assert results.get("/results/drift").status_code == 401
    assert results.get("/results/drift", headers={"Authorization": "Bearer wrong"}).status_code == 401
    # authorized but empty: the auth check passed and the lookup ran
    assert results.get("/results/drift", headers={"Authorization": "Bearer s3cret"}).status_code == 404
