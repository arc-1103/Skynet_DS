# src/github_integration/pr_client.py
"""GitHub-specific integration (Architecture.md github/: 'webhook parsing,
PR/commit diff fetching, code-change extraction'). Pulls a PR's metadata and
diff hunks from the GitHub REST API and normalizes them into schemas/pr.py
so nothing downstream needs to know about PyGithub.

Named github_integration/, not github/, because the latter would shadow the
installed PyGithub distribution's own top-level `github` module the moment
src/ ends up on sys.path (PYTHONPATH, an editable install, some pytest
configs) - `from github import Github` would then import this package
instead and fail."""

from github import Github

from src.schemas.pr import ChangedFile, PRPayload


def fetch_pr(repo_full_name: str, pr_number: int, token: str | None = None) -> PRPayload:
    """Fetch a pull request's metadata and per-file diff from GitHub."""
    client = Github(token) if token else Github()
    repo = client.get_repo(repo_full_name)
    pull = repo.get_pull(pr_number)

    changed_files = [
        ChangedFile(
            filename=f.filename,
            status=f.status,
            additions=f.additions,
            deletions=f.deletions,
            patch=f.patch,
        )
        for f in pull.get_files()
    ]

    return PRPayload(
        pr_id=pull.number,
        repo_full_name=repo_full_name,
        commit_sha=pull.head.sha,
        base_sha=pull.base.sha,
        base_ref=pull.base.ref,
        author=pull.user.login,
        title=pull.title,
        body=pull.body or "",
        changed_files=changed_files,
    )
