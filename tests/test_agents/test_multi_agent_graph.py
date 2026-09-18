"""End-to-end: the compiled LangGraph runs all specialist branches over one
PR and the synthesizer sees every branch's output. This is the actual
question a unit test per-agent can't answer - that operator.add correctly
merges concurrent writes instead of one branch clobbering another's."""

import subprocess

import numpy as np
import pandas as pd
import pytest

from src.agents.multi_agent_graph import build_pr_diff, run_pr_review
from src.schemas.pr import ChangedFile, PRPayload
from src.schemas.review import Severity


def _hunk(*added_lines: str) -> str:
    n = len(added_lines) + 1
    body = "".join(f"+{line}\n" for line in added_lines)
    return f"@@ -1,1 +1,{n} @@\n context\n{body}"


def _pr(changed_files, **overrides) -> PRPayload:
    defaults = dict(
        pr_id=7,
        repo_full_name="acme/demo",
        commit_sha="head",
        author="octocat",
        title="Mixed-issue PR",
        changed_files=changed_files,
    )
    defaults.update(overrides)
    return PRPayload(**defaults)


def test_build_pr_diff_includes_each_changed_files_patch():
    pr = _pr(
        [
            ChangedFile(filename="a.py", status="modified", patch=_hunk("x = 1")),
            ChangedFile(filename="b.py", status="added", patch=_hunk("y = 2")),
        ]
    )
    diff = build_pr_diff(pr)
    assert "a.py" in diff and "b.py" in diff
    assert "x = 1" in diff and "y = 2" in diff


def test_graph_merges_all_static_branches_without_repo(tmp_path):
    pr = _pr(
        [
            ChangedFile(
                filename="security.py",
                status="modified",
                patch=_hunk('api_key = "sk-1234567890abcd"'),
            ),
            ChangedFile(
                filename="train.py",
                status="modified",
                patch=_hunk("for i, row in df.iterrows():"),
            ),
            ChangedFile(
                filename="messy.py",
                status="modified",
                patch=_hunk("except:"),
            ),
        ]
    )

    final_state = run_pr_review(pr, repo_path=None)
    review = final_state["final_review"]

    assert final_state["drift_report"] == []
    titles = {f.title for f in review.inline_findings}
    assert "Hardcoded credential" in titles
    assert "Row-wise DataFrame iteration via .iterrows()" in titles
    assert "Bare except clause" in titles
    assert review.overall_passed is False  # hardcoded credential is CRITICAL


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
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")

    rng = np.random.default_rng(20260916)
    n = 200
    base = pd.DataFrame({"a": rng.normal(0, 1, n)})
    base_sha = _commit_frame(repo, base, "data/x.csv", "base")
    head = pd.DataFrame({"a": rng.normal(6, 1, n)})
    head_sha = _commit_frame(repo, head, "data/x.csv", "head")
    return repo, base_sha, head_sha


def test_graph_includes_drift_branch_when_repo_path_given(repo_with_drifted_csv):
    repo, base_sha, head_sha = repo_with_drifted_csv
    pr = _pr(
        [ChangedFile(filename="data/x.csv", status="modified")],
        commit_sha=head_sha,
        base_sha=base_sha,
    )

    final_state = run_pr_review(pr, repo_path=repo)

    assert len(final_state["drift_report"]) == 1
    assert final_state["drift_report"][0].dataset_drifted is True
    assert final_state["final_review"] is not None


def test_a_failing_synthesizer_does_not_discard_the_branches(monkeypatch):
    """Every branch was isolated but the fan-in wasn't, so one raise inside the
    synthesizer propagated out of app.invoke() and threw away all of it."""
    import src.agents.multi_agent_graph as graph_module

    def _boom(state):
        raise RuntimeError("synthesizer exploded")

    monkeypatch.setattr(graph_module, "run_synthesizer_agent", _boom)

    pr = _pr([ChangedFile(filename="src/x.py", status="modified", patch=_hunk("eval(x)"))])
    state = run_pr_review(pr)

    assert state["final_review"] is None
    assert any("synthesize" in e for e in state["errors"])
    assert state["security_findings"]  # the branch's work survived
