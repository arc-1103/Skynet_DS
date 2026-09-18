# demo/github_pr_comment.py
"""Posts the multi-agent graph's FinalPRReview back onto a real GitHub PR
as a review: a top-level summary comment (status badges + a findings
table, collapsed by default) plus one inline comment per finding that has
a line number.

This is the answer to "where is the UI?" for a demo - GitHub's own PR
screen already renders markdown tables, collapsible <details> sections,
and inline annotations; nothing here needs a hand-built frontend.

Deliberately kept out of src/: posting a review is a visible, external
side effect (it shows up on someone's real pull request) that shouldn't
happen just because the diagnostic core got imported, and it needs a
GITHUB_TOKEN with write access, which the read-only CLI/webhook path
(src/github_integration/pr_client.py) doesn't require. See
demo/README.md before running this against anything but a scratch repo."""

from __future__ import annotations

from github import Github

from src.schemas.review import FinalPRReview, ReviewFinding, Severity

_SEVERITY_BADGE_COLOR = {
    Severity.CRITICAL: "8B0000",  # dark red
    Severity.HIGH: "d73a49",  # red
    Severity.MEDIUM: "dbab09",  # yellow
    Severity.LOW: "0969da",  # blue
    Severity.INFO: "6e7781",  # grey
}

# GitHub's inline-comment API only accepts these for the "side" of a diff.
_DIFF_SIDE = "RIGHT"


def _badge(label: str, value: str, color: str) -> str:
    return f"![{label}](https://img.shields.io/badge/{label}-{value}-{color})"


def _status_badges(review: FinalPRReview) -> str:
    status_label = "PASSED" if review.overall_passed else "CHANGES_REQUESTED"
    status_color = "2ea44f" if review.overall_passed else "d73a49"
    badges = [_badge("review", status_label, status_color)]

    counts: dict[Severity, int] = {}
    for finding in review.inline_findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    for severity in Severity:
        if severity in counts:
            badges.append(_badge(severity.value, str(counts[severity]), _SEVERITY_BADGE_COLOR[severity]))
    return " ".join(badges)


def _findings_table(findings: list[ReviewFinding]) -> str:
    if not findings:
        return "_No issues found._\n"
    rows = ["| Severity | File | Line | Finding |", "|---|---|---|---|"]
    for f in findings:
        line = f.line_number if f.line_number is not None else "-"
        rows.append(f"| {f.severity.value} | `{f.file_path}` | {line} | {f.title} |")
    return "\n".join(rows) + "\n"


def _collapsible(summary: str, body: str) -> str:
    return f"<details>\n<summary>{summary}</summary>\n\n{body}\n</details>"


def build_summary_markdown(review: FinalPRReview, pr_title: str, pr_number: int) -> str:
    """The review's top-level body: title, status/severity badges, and a
    findings table collapsed behind a <summary> so the comment stays
    short until an evaluator clicks it open."""
    return "\n\n".join(
        [
            f"## \U0001f50e Automated PR Review — #{pr_number}: {pr_title}",
            _status_badges(review),
            _collapsible(f"Findings ({len(review.inline_findings)})", _findings_table(review.inline_findings)),
        ]
    )


def build_inline_comment(finding: ReviewFinding) -> str:
    """One finding's inline annotation body. Advice is rendered as a
    labeled blockquote rather than GitHub's literal ```suggestion``` diff
    block - that block implies an exact, one-click-appliable code
    replacement, and ReviewFinding.suggestion is prose guidance, not
    guaranteed-correct replacement source."""
    lines = [f"**[{finding.severity.value}] {finding.title}**", "", finding.description]
    if finding.suggestion:
        lines += ["", f"> \U0001f4a1 **Suggested fix:** {finding.suggestion}"]
    return "\n".join(lines)


def post_review_to_github(
    repo_full_name: str,
    pr_number: int,
    pr_title: str,
    review: FinalPRReview,
    token: str,
    *,
    event: str = "COMMENT",
) -> None:
    """Posts one GitHub review carrying both the summary comment and every
    inline-anchorable finding in a single API call. Findings without a
    line_number (e.g. architecture_agent's "large diff" check) only show
    up in the summary table - GitHub inline comments must anchor to a
    line, so there's nowhere else to put them.

    event defaults to "COMMENT" (never REQUEST_CHANGES/APPROVE) so running
    this doesn't silently gate-keep someone's PR merge without a human
    having chosen that on purpose."""
    client = Github(token)
    repo = client.get_repo(repo_full_name)
    pull = repo.get_pull(pr_number)
    commit = repo.get_commit(pull.head.sha)

    comments = [
        {
            "path": f.file_path,
            "line": f.line_number,
            "side": _DIFF_SIDE,
            "body": build_inline_comment(f),
        }
        for f in review.inline_findings
        if f.line_number is not None
    ]

    pull.create_review(
        commit=commit,
        body=build_summary_markdown(review, pr_title, pr_number),
        event=event,
        comments=comments,
    )
