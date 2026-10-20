# src/github_integration/fix_pr.py
"""Opens a "fix" pull request from a review's mechanical fixes.

Only findings carrying a LineFix are eligible - exact one-line patches that
deterministic checks produce (schemas/review.py). `suggestion` prose and LLM
findings are never applied: nothing here writes code a model made up.

Safety properties, all load-bearing:
  - The fix PR targets the author's PR *branch*, not the default branch, so
    a human reviews it and merging it is the author's call. Nothing here
    merges, approves, or touches main.
  - A fix is applied only if the file at the PR's head commit still has the
    exact `old_line` at that line number; anything else is skipped.
  - Fork PRs are skipped (no write access to the fork), and so are fix PRs
    themselves, so a fix PR's own review can't spawn a fix of the fix.
  - Off unless MLCOP_OPEN_FIX_PRS says otherwise; "dry-run" plans and logs
    the patch without writing anything to GitHub.

Needs a token that can push a branch and open a PR (contents: write,
pull_requests: write). The commits and PR are authored by whoever owns that
token - an app identity is still an open decision (see jobs.py)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from github import Github, GithubException

from src.schemas.review import FinalPRReview, LineFix

logger = logging.getLogger(__name__)

BRANCH_PREFIX = "skynet/fix-"


@dataclass
class FixPRResult:
    branch: str
    applied: list[tuple[str, int, LineFix]] = field(default_factory=list)  # (path, line, fix)
    skipped: list[tuple[str, int, str]] = field(default_factory=list)  # (path, line, why)
    url: str | None = None  # None for a dry run


def apply_line_fixes(content: str, fixes: list[tuple[int, LineFix]]) -> tuple[str, list[int], list[tuple[int, str]]]:
    """Apply one-line fixes to a file's text. Returns (new text, applied line
    numbers, [(skipped line number, reason)]). Splits on "\\n" only, so a
    CRLF file's "\\r" stays on each line and compares equal to old_line."""
    lines = content.split("\n")
    applied: list[int] = []
    skipped: list[tuple[int, str]] = []
    for line_no, fix in sorted(fixes, key=lambda f: f[0]):
        if not 1 <= line_no <= len(lines):
            skipped.append((line_no, "line is past the end of the file"))
        elif lines[line_no - 1] != fix.old_line:
            skipped.append((line_no, "line no longer matches what was reviewed"))
        else:
            lines[line_no - 1] = fix.new_line
            applied.append(line_no)
    return "\n".join(lines), applied, skipped


def _eligible(review: FinalPRReview) -> dict[str, list[tuple[int, LineFix]]]:
    by_file: dict[str, dict[int, LineFix]] = {}
    for f in review.inline_findings:
        if f.fix and f.line_number is not None:
            # two fixes on one line would each be written against the other's
            # old_line; first wins, the second shows up as already-handled
            by_file.setdefault(f.file_path, {}).setdefault(f.line_number, f.fix)
    return {path: sorted(fixes.items()) for path, fixes in by_file.items()}


def open_fix_pr(
    repo_full_name: str,
    pr_number: int,
    review: FinalPRReview,
    token: str,
    *,
    dry_run: bool = False,
) -> FixPRResult | None:
    """Returns None when there is nothing to do (no eligible fix, PR not
    eligible, or a fix branch for this head commit already exists).

    Every call also retires this PR's fix PRs for older head commits, even
    when there is nothing new to fix: a push that resolves the issues
    should not leave the old fix PR open."""
    eligible = _eligible(review)
    repo = Github(token).get_repo(repo_full_name)
    pull = repo.get_pull(pr_number)
    if pull.state != "open":
        logger.info("PR #%s is %s - no fix PR", pr_number, pull.state)
        return None
    if pull.head.ref.startswith(BRANCH_PREFIX):
        logger.info("PR #%s is itself a fix PR - not fixing the fix", pr_number)
        return None
    if pull.head.repo is None or pull.head.repo.full_name.lower() != repo_full_name.lower():
        logger.info("PR #%s comes from a fork - no write access, no fix PR", pr_number)
        return None

    head_sha = pull.head.sha
    branch = f"{BRANCH_PREFIX}pr{pr_number}-{head_sha[:7]}"
    if not dry_run:
        _close_superseded(repo, pr_number, keep=branch)
    if not eligible:
        return None
    result = FixPRResult(branch=branch)
    patched: list[tuple[str, str, str]] = []  # (path, blob sha, new text)

    for path, fixes in eligible.items():
        try:
            blob = repo.get_contents(path, ref=head_sha)
            text = blob.decoded_content.decode("utf-8")
        except (GithubException, UnicodeDecodeError, AttributeError):  # missing, binary, or a directory
            result.skipped += [(path, line, "file could not be read as text at the PR head") for line, _ in fixes]
            continue
        new_text, applied, skipped = apply_line_fixes(text, fixes)
        by_line = dict(fixes)
        result.applied += [(path, line, by_line[line]) for line in applied]
        result.skipped += [(path, line, why) for line, why in skipped]
        if applied:
            patched.append((path, blob.sha, new_text))

    if not patched:
        logger.info("PR #%s: no fix still applies at %s", pr_number, head_sha[:7])
        return None
    if dry_run:
        for path, line, fix in result.applied:
            logger.info("[dry-run] fix PR %s: %s:%s\n  - %s\n  + %s", branch, path, line, fix.old_line, fix.new_line)
        return result

    try:
        repo.get_git_ref(f"heads/{branch}")
        logger.info("fix branch %s already exists - not opening another", branch)
        return None
    except GithubException as exc:
        if exc.status != 404:
            raise

    repo.create_git_ref(f"refs/heads/{branch}", head_sha)
    try:
        for path, blob_sha, new_text in patched:
            n = sum(1 for p, _, _ in result.applied if p == path)
            repo.update_file(
                path,
                f"fix: apply {n} automated review fix{'es' if n != 1 else ''} to {path}",
                new_text,
                blob_sha,
                branch=branch,
            )
        # base is the PR's own branch: the author merges (or closes) this
        created = repo.create_pull(
            title=f"Automated fixes for #{pr_number}",
            body=_body(pr_number, result),
            base=pull.head.ref,
            head=branch,
        )
    except Exception:
        # don't leave a half-written branch that would block a retry
        try:
            repo.get_git_ref(f"heads/{branch}").delete()
        except GithubException:
            logger.warning("could not clean up partial fix branch %s", branch, exc_info=True)
        raise
    result.url = created.html_url
    logger.info("opened fix PR %s for #%s", result.url, pr_number)
    return result


def _close_superseded(repo, pr_number: int, keep: str) -> None:
    """Close this PR's open fix PRs whose branch is for an older head commit,
    and delete those branches. Matches only our own `skynet/fix-pr<N>-`
    branches in this repo, so nothing a human opened is ever touched."""
    prefix = f"{BRANCH_PREFIX}pr{pr_number}-"
    for fix in repo.get_pulls(state="open"):
        ref = fix.head.ref
        if not ref.startswith(prefix) or ref == keep or fix.head.repo is None:
            continue
        if fix.head.repo.full_name.lower() != repo.full_name.lower():
            continue
        try:
            fix.create_issue_comment(f"Superseded: PR #{pr_number} has new commits, so these fixes were regenerated against them.")
            fix.edit(state="closed")
            repo.get_git_ref(f"heads/{ref}").delete()
            logger.info("closed superseded fix PR #%s (%s)", fix.number, ref)
        except GithubException:
            logger.warning("could not retire superseded fix PR #%s", fix.number, exc_info=True)


def _body(pr_number: int, result: FixPRResult) -> str:
    intro = (
        f"Mechanical fixes for the automated review of #{pr_number}. "
        "Each is an exact one-line change from a deterministic check, applied only because the line "
        "still matched what was reviewed. **Nothing here is merged for you** - review it, merge it "
        "into the PR branch, or close it."
    )
    lines = [
        intro,
        "",
    ]
    for path, line, fix in result.applied:
        lines += [f"**`{path}:{line}`**", "```diff", f"- {fix.old_line.strip()}", f"+ {fix.new_line.strip()}", "```"]
    if result.skipped:
        lines += ["", "<details><summary>Skipped</summary>", ""]
        lines += [f"- `{path}:{line}` - {why}" for path, line, why in result.skipped]
        lines += ["", "</details>"]
    return "\n".join(lines)
