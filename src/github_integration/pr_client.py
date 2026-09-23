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

from datetime import datetime

from github import Github

from src.schemas.pr import ChangedFile, PRPayload


_MAX_FILES_LISTED = 5


def fetch_commit_diff(repo_full_name: str, base_ref: str, head_ref: str, token: str | None = None) -> str:
    """One-line summary of what changed between two commits - the code-diff
    evidence for a diagnosis (Workflow 1 step 4, Workflow 4 step 3)."""
    client = Github(token) if token else Github()
    comparison = client.get_repo(repo_full_name).compare(base_ref, head_ref)
    files = list(comparison.files)
    added = sum(f.additions for f in files)
    removed = sum(f.deletions for f in files)
    names = ", ".join(f.filename for f in files[:_MAX_FILES_LISTED])
    if len(files) > _MAX_FILES_LISTED:
        names += f", +{len(files) - _MAX_FILES_LISTED} more"
    return (
        f"{comparison.total_commits} commit(s), {len(files)} file(s) changed "
        f"(+{added}/-{removed}): {names or 'no files'}"
    )


def fetch_path_commits(
    repo_full_name: str, paths: tuple[str, ...], since: datetime, token: str | None = None
) -> list[str]:
    """Commits since `since` touching any of `paths`, newest first, as
    "<short sha> <first message line>" - the pipeline-change evidence for a
    drift (Workflow 2 step 4)."""
    repo = (Github(token) if token else Github()).get_repo(repo_full_name)
    seen: dict[str, tuple[datetime, str]] = {}
    for path in paths:
        for commit in repo.get_commits(since=since, path=path):
            message = commit.commit.message.splitlines()[0] if commit.commit.message else ""
            seen[commit.sha] = (commit.commit.author.date, f"{commit.sha[:7]} {message}")
    return [label for _, label in sorted(seen.values(), reverse=True)]


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
