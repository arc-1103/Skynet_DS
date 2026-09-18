import json
import subprocess

import pytest

from src.agents import drift_agent
from src.core.config import settings
from src.schemas.drift import DatasetDriftReport
from src.schemas.pr import ChangedFile, PRPayload
from src.worker import jobs


@pytest.mark.asyncio
async def test_run_deployment_regression_check_diagnoses_seeded_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # memory.db lands in tmp_path, not the repo

    await jobs.run_deployment_regression_check(
        repo_full_name="acme/fraud-clf",
        deploy_sha="abc123",
        environment="production",
    )
    # No assertion beyond "it completes without raising" - this job is a
    # documented stub (see TODO in src/worker/jobs.py) until Phase 1 wires a
    # real repo -> model_id mapping and a real adapter.


@pytest.mark.asyncio
async def test_run_deployment_regression_check_handles_unknown_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(jobs, "_STUB_MODEL_ID", "no-such-model")

    await jobs.run_deployment_regression_check(
        repo_full_name="acme/other",
        deploy_sha="def456",
        environment="staging",
    )
    # Should log a warning and return cleanly, not raise.


@pytest.mark.asyncio
async def test_run_pr_code_review_caches_fetched_pr_locally(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    fake_pr = PRPayload(
        pr_id=42,
        repo_full_name="acme/fraud-clf",
        commit_sha="deadbeef",
        author="octocat",
        title="Fix flaky test",
        body="Closes #41",
        changed_files=[ChangedFile(filename="tests/test_x.py", status="modified", patch="@@ -1 +1 @@")],
    )
    captured = {}

    def fake_fetch_pr(repo_full_name, pr_number, token=None):
        captured["args"] = (repo_full_name, pr_number, token)
        return fake_pr

    monkeypatch.setattr(jobs, "fetch_pr", fake_fetch_pr)

    await jobs.run_pr_code_review(
        repo_full_name="acme/fraud-clf",
        pr_number=42,
        head_sha="deadbeef",
    )

    assert captured["args"] == ("acme/fraud-clf", 42, settings.github_token)

    out_path = tmp_path / settings.pr_cache_dir / "acme_fraud-clf" / "pr_42_deadbeef.json"
    assert out_path.exists()
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert saved["pr_id"] == 42
    assert saved["changed_files"][0]["filename"] == "tests/test_x.py"


@pytest.mark.asyncio
async def test_run_pr_code_review_skips_drift_for_a_code_only_pr(tmp_path, monkeypatch):
    """Drift checking needs a clone, so a PR touching no data files must not
    trigger one."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        jobs,
        "fetch_pr",
        lambda *a, **k: PRPayload(
            pr_id=7,
            repo_full_name="acme/fraud-clf",
            commit_sha="cafe",
            base_sha="beef",
            author="octocat",
            title="Code only",
            changed_files=[ChangedFile(filename="src/model.py", status="modified")],
        ),
    )
    monkeypatch.setattr(
        jobs, "clone_or_checkout", lambda *a, **k: pytest.fail("should not clone")
    )

    await jobs.run_pr_code_review(repo_full_name="acme/fraud-clf", pr_number=7, head_sha="cafe")


@pytest.mark.asyncio
async def test_run_pr_code_review_writes_a_drift_report_per_data_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        jobs,
        "fetch_pr",
        lambda *a, **k: PRPayload(
            pr_id=8,
            repo_full_name="acme/fraud-clf",
            commit_sha="cafe",
            base_sha="beef",
            author="octocat",
            title="Refresh data",
            changed_files=[ChangedFile(filename="data/train.csv", status="modified")],
        ),
    )
    monkeypatch.setattr(jobs, "clone_or_checkout", lambda *a, **k: tmp_path / "clone")
    # drift now reaches the job through the graph's drift branch, so the stub
    # goes where that branch imported it
    monkeypatch.setattr(
        drift_agent,
        "check_pr_drift",
        lambda pr, repo_path: [
            DatasetDriftReport(columns=[], share_threshold=0.5, label="data/train.csv")
        ],
    )

    await jobs.run_pr_code_review(repo_full_name="acme/fraud-clf", pr_number=8, head_sha="cafe")

    cache_dir = tmp_path / settings.pr_cache_dir / "acme_fraud-clf"
    drift_path = cache_dir / "pr_8_cafe_drift_data_train.csv.json"
    assert drift_path.exists()
    assert json.loads(drift_path.read_text(encoding="utf-8"))["label"] == "data/train.csv"

    review = json.loads((cache_dir / "pr_8_cafe_review.json").read_text(encoding="utf-8"))
    assert review["overall_passed"] is True


@pytest.mark.asyncio
async def test_run_pr_code_review_writes_the_multi_agent_review(tmp_path, monkeypatch):
    """The graph used to be unreachable from the webhook path entirely."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        jobs,
        "fetch_pr",
        lambda *a, **k: PRPayload(
            pr_id=9,
            repo_full_name="acme/fraud-clf",
            commit_sha="cafe",
            author="octocat",
            title="Add handler",
            changed_files=[
                ChangedFile(
                    filename="src/handler.py",
                    status="modified",
                    patch="@@ -1,1 +1,2 @@\n context\n+eval(user_input)\n",
                )
            ],
        ),
    )
    monkeypatch.setattr(
        jobs, "clone_or_checkout", lambda *a, **k: pytest.fail("no data file - should not clone")
    )

    await jobs.run_pr_code_review(repo_full_name="acme/fraud-clf", pr_number=9, head_sha="cafe")

    review_path = tmp_path / settings.pr_cache_dir / "acme_fraud-clf" / "pr_9_cafe_review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))

    assert review["overall_passed"] is False
    assert [f["title"] for f in review["inline_findings"]] == ["Use of eval()"]
    assert review["inline_findings"][0]["line_number"] == 2


@pytest.mark.asyncio
async def test_run_pr_code_review_rejects_a_head_sha_that_is_not_a_sha(tmp_path, monkeypatch):
    """head_sha comes straight off the webhook payload and is interpolated
    into the cache filename, so it must not be able to carry a path."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(jobs, "fetch_pr", lambda *a, **k: pytest.fail("should not fetch"))

    with pytest.raises(ValueError, match="non-SHA head_sha"):
        await jobs.run_pr_code_review(
            repo_full_name="acme/fraud-clf", pr_number=9, head_sha="../../../../x"
        )


def _data_pr(pr_id: int):
    return PRPayload(
        pr_id=pr_id,
        repo_full_name="acme/fraud-clf",
        commit_sha="cafe",
        base_sha="beef",
        author="octocat",
        title="Refresh data",
        changed_files=[ChangedFile(filename="data/train.csv", status="modified")],
    )


@pytest.mark.asyncio
async def test_run_pr_code_review_still_reviews_when_the_clone_fails(tmp_path, monkeypatch):
    """A transient git failure shouldn't cost the whole review - the PR is
    already cached and the static branches work off the diff alone."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(jobs, "fetch_pr", lambda *a, **k: _data_pr(10))

    def _git_exploded(*a, **k):
        raise subprocess.CalledProcessError(128, ["git", "clone"])

    monkeypatch.setattr(jobs, "clone_or_checkout", _git_exploded)

    await jobs.run_pr_code_review(repo_full_name="acme/fraud-clf", pr_number=10, head_sha="cafe")

    cache_dir = tmp_path / settings.pr_cache_dir / "acme_fraud-clf"
    review = json.loads((cache_dir / "pr_10_cafe_review.json").read_text(encoding="utf-8"))
    assert review["overall_passed"] is True
    assert not list(cache_dir.glob("*_drift_*.json"))


@pytest.mark.asyncio
async def test_drift_report_filename_survives_a_backslash_in_the_label(tmp_path, monkeypatch):
    """git tracks paths containing a literal backslash, and with_name() raises
    on an OS separator - which on Windows silently lost every remaining drift
    report for that PR."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(jobs, "fetch_pr", lambda *a, **k: _data_pr(11))
    monkeypatch.setattr(jobs, "clone_or_checkout", lambda *a, **k: tmp_path / "clone")
    monkeypatch.setattr(
        drift_agent,
        "check_pr_drift",
        lambda pr, repo_path: [
            DatasetDriftReport(columns=[], share_threshold=0.5, label="data\\odd:name.csv")
        ],
    )

    await jobs.run_pr_code_review(repo_full_name="acme/fraud-clf", pr_number=11, head_sha="cafe")

    cache_dir = tmp_path / settings.pr_cache_dir / "acme_fraud-clf"
    written = list(cache_dir.glob("pr_11_cafe_drift_*.json"))
    assert len(written) == 1
    assert written[0].name == "pr_11_cafe_drift_data_odd_name.csv.json"
