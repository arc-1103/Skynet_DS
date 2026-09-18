"""post_review_to_github itself is a thin PyGithub I/O wrapper (same
convention as src/github_integration/pr_client.py and repo_client.py -
neither has unit tests either, since there's no business logic to test
without a real network call). What's worth testing here is the pure
markdown-building: badges, the findings table, and the inline comment
body - that's what an evaluator actually sees on the PR."""

from demo.github_pr_comment import (
    build_inline_comment,
    build_summary_markdown,
    _findings_table,
    _status_badges,
)
from src.schemas.review import FinalPRReview, ReviewFinding, Severity


def _finding(**overrides) -> ReviewFinding:
    defaults = dict(
        file_path="a.py",
        line_number=10,
        title="Some issue",
        severity=Severity.HIGH,
        description="It's a problem.",
        suggestion=None,
    )
    defaults.update(overrides)
    return ReviewFinding(**defaults)


def test_status_badge_is_green_when_passed():
    review = FinalPRReview(overall_passed=True, markdown_summary="", inline_findings=[])
    badges = _status_badges(review)
    assert "PASSED-2ea44f" in badges
    assert "CHANGES_REQUESTED" not in badges


def test_status_badge_is_red_when_failed():
    review = FinalPRReview(overall_passed=False, markdown_summary="", inline_findings=[])
    assert "CHANGES_REQUESTED-d73a49" in _status_badges(review)


def test_status_badges_include_one_per_severity_present():
    review = FinalPRReview(
        overall_passed=False,
        markdown_summary="",
        inline_findings=[_finding(severity=Severity.HIGH), _finding(severity=Severity.LOW)],
    )
    badges = _status_badges(review)
    assert "HIGH-1" in badges
    assert "LOW-1" in badges
    assert "CRITICAL" not in badges


def test_findings_table_lists_each_finding():
    findings = [_finding(file_path="x.py", line_number=5, title="Bug A")]
    table = _findings_table(findings)
    assert "x.py" in table
    assert "5" in table
    assert "Bug A" in table


def test_findings_table_empty_case():
    assert "No issues found" in _findings_table([])


def test_findings_table_handles_missing_line_number():
    findings = [_finding(line_number=None, title="No line")]
    table = _findings_table(findings)
    assert "| - |" in table


def test_build_summary_markdown_includes_title_and_pr_number():
    review = FinalPRReview(overall_passed=True, markdown_summary="", inline_findings=[])
    md = build_summary_markdown(review, "Fix the bug", 7)
    assert "#7" in md
    assert "Fix the bug" in md
    assert "<details>" in md and "</details>" in md


def test_build_inline_comment_includes_suggestion_when_present():
    finding = _finding(suggestion="Use X instead of Y.")
    body = build_inline_comment(finding)
    assert "Suggested fix" in body
    assert "Use X instead of Y." in body


def test_build_inline_comment_omits_suggestion_block_when_absent():
    finding = _finding(suggestion=None)
    body = build_inline_comment(finding)
    assert "Suggested fix" not in body
