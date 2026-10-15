"""The shared LLM-findings plumbing the three specialist agents delegate to."""

import threading

import pytest

from src.agents.llm_findings import collect_llm_findings, parse_findings
from src.schemas.pr import ChangedFile
from src.schemas.review import Severity


def _file(name: str, added: str = "x = 1") -> ChangedFile:
    return ChangedFile(
        filename=name, status="modified", patch=f"@@ -1,1 +1,2 @@\n context\n+{added}\n"
    )


def test_parse_findings_builds_a_review_finding_per_item():
    findings = parse_findings(
        "a.py",
        [{"line_number": 3, "title": "t", "severity": "high", "description": "d", "suggestion": "s"}],
    )
    assert len(findings) == 1
    assert findings[0].file_path == "a.py"
    assert findings[0].severity is Severity.HIGH


@pytest.mark.parametrize(
    "item",
    [
        {"title": "t"},  # no description
        {"description": "d"},  # no title
        {"title": "t", "description": "d", "severity": "CATASTROPHIC"},  # not one of ours
    ],
)
def test_parse_findings_skips_items_it_cannot_build(item):
    assert parse_findings("a.py", [item]) == []


def test_parse_findings_treats_an_explicit_null_severity_as_medium():
    """The model returns `null` rather than omitting the key, so `.get(k, default)`
    would hand None straight to Severity()."""
    findings = parse_findings("a.py", [{"title": "t", "description": "d", "severity": None}])
    assert findings[0].severity is Severity.MEDIUM


def test_parse_findings_tolerates_none():
    assert parse_findings("a.py", None) == []


def test_parse_findings_drops_an_out_of_diff_line_number_but_keeps_the_finding():
    """GitHub's review API rejects the *entire* review if one inline
    comment's line isn't in the diff, so a model-reported line_number
    outside valid_lines must not reach github_pr_comment.py as-is - but
    the finding itself is still real content, so it's kept with
    line_number cleared rather than dropped."""
    findings = parse_findings(
        "a.py",
        [{"line_number": 999, "title": "t", "description": "d"}],
        valid_lines={1, 2, 3},
    )
    assert len(findings) == 1
    assert findings[0].line_number is None


def test_parse_findings_keeps_a_line_number_inside_valid_lines():
    findings = parse_findings(
        "a.py", [{"line_number": 2, "title": "t", "description": "d"}], valid_lines={1, 2, 3}
    )
    assert findings[0].line_number == 2


def test_collect_llm_findings_asks_once_per_file_and_keeps_file_order():
    asked = []

    def ask(system, user):
        asked.append(user.splitlines()[0])
        name = user.split("File: ", 1)[1].splitlines()[0]
        return [{"title": f"from {name}", "description": "d"}]

    files = [_file("a.py"), _file("b.py"), _file("c.py")]
    findings = collect_llm_findings(files, "system", ask)

    assert [f.file_path for f in findings] == ["a.py", "b.py", "c.py"]
    assert len(asked) == 3


def test_collect_llm_findings_drops_a_line_number_outside_the_diff():
    # _file("a.py")'s only added line lands at new-file line 2 (see its patch).
    findings = collect_llm_findings(
        [_file("a.py")], "system", lambda s, u: [{"line_number": 999, "title": "t", "description": "d"}]
    )
    assert findings[0].line_number is None


def test_collect_llm_findings_skips_files_with_no_added_lines():
    called = []
    files = [
        ChangedFile(filename="empty.py", status="modified", patch=None),
        ChangedFile(filename="no_adds.py", status="modified", patch="@@ -1,1 +1,1 @@\n ctx\n"),
    ]
    collect_llm_findings(files, "system", lambda s, u: called.append(u) or [])
    assert called == []


def test_collect_llm_findings_runs_the_per_file_calls_concurrently():
    """These are independent network round trips; a 20-file PR used to mean 20
    of them back to back on a webhook job's critical path. The barrier only
    clears if all four calls are genuinely in flight at once - run them
    sequentially and it times out."""
    barrier = threading.Barrier(4, timeout=10)

    def ask(system, user):
        barrier.wait()
        return []

    collect_llm_findings([_file(f"f{i}.py") for i in range(4)], "system", ask)
    assert not barrier.broken
