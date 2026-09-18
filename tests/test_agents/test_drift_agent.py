"""drift_agent wraps check_pr_drift as a LangGraph node - these tests cover
the wiring (state in/out, error handling), not the drift math itself
(covered by tests/test_pr_drift.py)."""

import subprocess

import numpy as np
import pandas as pd
import pytest

from src.agents.drift_agent import make_drift_agent
from src.schemas.pr import ChangedFile, PRPayload

SEED = 20260916


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _commit_frame(repo, frame: pd.DataFrame, path: str, message: str) -> str:
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(repo / path, index=False)
    _git(repo, "add", path)
    _git(repo, "commit", "-m", message)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo_with_csv(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")

    rng = np.random.default_rng(SEED)
    n = 200
    base = pd.DataFrame({"a": rng.normal(0, 1, n)})
    base_sha = _commit_frame(repo, base, "data/x.csv", "base")
    head = pd.DataFrame({"a": rng.normal(5, 1, n)})
    head_sha = _commit_frame(repo, head, "data/x.csv", "head")
    return repo, base_sha, head_sha


def _pr(base_sha, head_sha, filename="data/x.csv"):
    return PRPayload(
        pr_id=1,
        repo_full_name="acme/demo",
        commit_sha=head_sha,
        base_sha=base_sha,
        author="octocat",
        title="Update data",
        changed_files=[ChangedFile(filename=filename, status="modified")],
    )


def test_drift_agent_returns_reports_under_drift_report_key(repo_with_csv):
    repo, base_sha, head_sha = repo_with_csv
    agent = make_drift_agent(repo)
    result = agent({"pr_payload": _pr(base_sha, head_sha)})
    assert len(result["drift_report"]) == 1
    assert result["drift_report"][0].label == "data/x.csv"


def test_drift_agent_handles_missing_base_sha_without_raising():
    pr = PRPayload(
        pr_id=1,
        repo_full_name="acme/demo",
        commit_sha="head",
        author="octocat",
        title="No base",
        changed_files=[ChangedFile(filename="data/x.csv", status="modified")],
    )
    agent = make_drift_agent("/nonexistent/repo")
    result = agent({"pr_payload": pr})
    assert result["drift_report"] == []
    assert any("base_sha" in e for e in result["errors"])


def test_drift_agent_empty_for_code_only_pr(repo_with_csv):
    repo, base_sha, head_sha = repo_with_csv
    pr = PRPayload(
        pr_id=1,
        repo_full_name="acme/demo",
        commit_sha=head_sha,
        base_sha=base_sha,
        author="octocat",
        title="Code change",
        changed_files=[ChangedFile(filename="src/model.py", status="modified")],
    )
    agent = make_drift_agent(repo)
    result = agent({"pr_payload": pr})
    assert result["drift_report"] == []
