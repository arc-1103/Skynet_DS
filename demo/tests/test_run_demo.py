"""run_demo.py's own glue code: this is the one place in the demo that
calls clone_or_checkout directly, so it's also the one place that can
forget the extra_refs the drift/dashboard comparisons below depend on."""

from src.schemas.pr import PRPayload
from src.schemas.review import FinalPRReview

import demo.run_demo as run_demo


def _pr(base_sha: str | None) -> PRPayload:
    return PRPayload(
        pr_id=42,
        repo_full_name="acme/demo",
        commit_sha="a" * 40,
        base_sha=base_sha,
        author="octocat",
        title="t",
        changed_files=[],
    )


def test_clone_or_checkout_is_given_the_pr_base_sha_as_an_extra_ref(monkeypatch):
    """Without this, an existing cached clone only fetches pr.commit_sha -
    if the base branch has since moved, `git show <base_sha>:<path>`
    (used by both the drift branch and the Evidently dashboards) fails to
    find it locally."""
    monkeypatch.setattr(run_demo, "fetch_pr", lambda *a, **k: _pr(base_sha="b" * 40))
    calls = []
    monkeypatch.setattr(
        run_demo, "clone_or_checkout", lambda *a, **k: calls.append((a, k)) or "repo_path"
    )
    monkeypatch.setattr(
        run_demo,
        "run_traced_pr_review",
        lambda pr, repo_path: {"final_review": FinalPRReview(markdown_summary="ok", overall_passed=True)},
    )
    monkeypatch.setattr(run_demo, "dashboards_for_pr", lambda pr, repo_path: [])
    monkeypatch.setattr(run_demo.sys, "argv", ["run_demo", "acme/demo", "42"])

    run_demo.main()

    assert calls[0][1]["extra_refs"] == ("b" * 40,)


def test_extra_refs_is_empty_when_the_pr_has_no_base_sha(monkeypatch):
    monkeypatch.setattr(run_demo, "fetch_pr", lambda *a, **k: _pr(base_sha=None))
    calls = []
    monkeypatch.setattr(
        run_demo, "clone_or_checkout", lambda *a, **k: calls.append((a, k)) or "repo_path"
    )
    monkeypatch.setattr(
        run_demo,
        "run_traced_pr_review",
        lambda pr, repo_path: {"final_review": FinalPRReview(markdown_summary="ok", overall_passed=True)},
    )
    monkeypatch.setattr(run_demo, "dashboards_for_pr", lambda pr, repo_path: [])
    monkeypatch.setattr(run_demo.sys, "argv", ["run_demo", "acme/demo", "42"])

    run_demo.main()

    assert calls[0][1]["extra_refs"] == ()
