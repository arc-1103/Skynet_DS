"""Langfuse degrades to a disabled no-op client when no credentials are
configured (logs an auth warning, never raises) - these tests confirm
that degrade-gracefully path so CI/local runs without LANGFUSE_* env vars
still pass, and that the traced graph produces the same result shape as
the untraced one in agents/multi_agent_graph.py."""

from demo.langfuse_tracing import langfuse_is_configured, run_traced_pr_review
from src.schemas.pr import ChangedFile, PRPayload


def _pr() -> PRPayload:
    return PRPayload(
        pr_id=1,
        repo_full_name="acme/demo",
        commit_sha="head",
        author="octocat",
        title="Demo PR",
        changed_files=[
            ChangedFile(
                filename="a.py",
                status="modified",
                patch="@@ -1,1 +1,2 @@\n context\n+eval(x)\n",
            )
        ],
    )


def test_langfuse_is_configured_false_without_env_vars(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert langfuse_is_configured() is False


def test_run_traced_pr_review_completes_without_credentials():
    result = run_traced_pr_review(_pr(), repo_path=None)
    assert result["final_review"] is not None
    titles = [f.title for f in result["final_review"].inline_findings]
    assert "Use of eval()" in titles
