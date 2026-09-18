"""PR-level drift: picking the comparable files out of a PR, reading both
versions of each straight from git, and reporting per-file drift.

These run against a real temporary git repository rather than a mocked one
- the whole point of load_at_ref is that `git show <sha>:<path>` behaves,
which a mock would assert nothing about."""

import subprocess

import numpy as np
import pandas as pd
import pytest

from src.drift.pr_drift import check_pr_drift, data_files_in_pr, load_at_ref
from src.schemas.pr import ChangedFile, PRPayload

SEED = 20260915


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
def repo_with_drifted_csv(tmp_path):
    """A repo whose second commit shifts one column and leaves another alone."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")

    rng = np.random.default_rng(SEED)
    n = 300
    base = pd.DataFrame(
        {"stable": rng.normal(5.0, 1.0, n), "shifting": rng.normal(5.0, 1.0, n)}
    )
    base_sha = _commit_frame(repo, base, "data/train.csv", "add training data")

    head = pd.DataFrame(
        {"stable": rng.normal(5.0, 1.0, n), "shifting": rng.normal(12.0, 1.0, n)}
    )
    head_sha = _commit_frame(repo, head, "data/train.csv", "refresh training data")
    return repo, base_sha, head_sha


def _payload(base_sha: str, head_sha: str, files: list[ChangedFile]) -> PRPayload:
    return PRPayload(
        pr_id=1,
        repo_full_name="acme/demo",
        commit_sha=head_sha,
        base_sha=base_sha,
        base_ref="main",
        author="octocat",
        title="Refresh training data",
        changed_files=files,
    )


def test_data_files_in_pr_picks_only_comparable_tabular_files():
    pr = _payload(
        "base",
        "head",
        [
            ChangedFile(filename="data/train.csv", status="modified"),
            ChangedFile(filename="data/features.parquet", status="modified"),
            ChangedFile(filename="src/model.py", status="modified"),
            ChangedFile(filename="README.md", status="modified"),
            # No baseline to compare against on either of these.
            ChangedFile(filename="data/new.csv", status="added"),
            ChangedFile(filename="data/old.csv", status="removed"),
        ],
    )
    assert data_files_in_pr(pr) == ["data/train.csv", "data/features.parquet"]


def test_data_files_in_pr_is_empty_for_a_code_only_pr():
    pr = _payload("base", "head", [ChangedFile(filename="src/api.py", status="modified")])
    assert data_files_in_pr(pr) == []


def test_load_at_ref_reads_each_version_of_the_file(repo_with_drifted_csv):
    repo, base_sha, head_sha = repo_with_drifted_csv
    base = load_at_ref(repo, base_sha, "data/train.csv")
    head = load_at_ref(repo, head_sha, "data/train.csv")
    assert list(base.columns) == ["stable", "shifting"]
    assert base["shifting"].mean() == pytest.approx(5.0, abs=0.5)
    assert head["shifting"].mean() == pytest.approx(12.0, abs=0.5)


def test_check_pr_drift_flags_the_shifted_column(repo_with_drifted_csv):
    repo, base_sha, head_sha = repo_with_drifted_csv
    pr = _payload(base_sha, head_sha, [ChangedFile(filename="data/train.csv", status="modified")])

    reports = check_pr_drift(pr, repo)

    assert len(reports) == 1
    report = reports[0]
    assert report.label == "data/train.csv"
    drifted = {c.column for c in report.columns if c.drifted}
    assert drifted == {"shifting"}
    assert report.share_drifted == pytest.approx(0.5)


def test_check_pr_drift_returns_nothing_for_a_code_only_pr(repo_with_drifted_csv):
    repo, base_sha, head_sha = repo_with_drifted_csv
    pr = _payload(base_sha, head_sha, [ChangedFile(filename="src/model.py", status="modified")])
    assert check_pr_drift(pr, repo) == []


def test_check_pr_drift_needs_a_base_to_compare_against(repo_with_drifted_csv):
    repo, _, head_sha = repo_with_drifted_csv
    pr = PRPayload(
        pr_id=1,
        repo_full_name="acme/demo",
        commit_sha=head_sha,
        author="octocat",
        title="No base recorded",
        changed_files=[ChangedFile(filename="data/train.csv", status="modified")],
    )
    with pytest.raises(ValueError, match="base_sha"):
        check_pr_drift(pr, repo)


def test_unreadable_file_is_skipped_not_fatal(repo_with_drifted_csv):
    """A file git cannot produce at that ref should drop out of the report
    rather than sink the whole PR check."""
    repo, base_sha, head_sha = repo_with_drifted_csv
    pr = _payload(
        base_sha,
        head_sha,
        [
            ChangedFile(filename="data/does_not_exist.csv", status="modified"),
            ChangedFile(filename="data/train.csv", status="modified"),
        ],
    )
    reports = check_pr_drift(pr, repo)
    assert [r.label for r in reports] == ["data/train.csv"]
