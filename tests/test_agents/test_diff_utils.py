from src.agents.diff_utils import added_text, dedupe_findings, iter_added_lines, numbered_added_lines
from src.schemas.review import ReviewFinding, Severity


def test_iter_added_lines_numbers_against_the_new_file():
    patch = (
        "@@ -1,3 +1,4 @@\n"
        " import pandas as pd\n"
        "+df.iterrows()\n"
        " x = 1\n"
        " y = 2\n"
    )
    assert list(iter_added_lines(patch)) == [(2, "df.iterrows()")]


def test_iter_added_lines_skips_removed_lines_without_numbering_them():
    patch = (
        "@@ -1,2 +1,2 @@\n"
        "-old_call()\n"
        "+new_call()\n"
        " unchanged\n"
    )
    assert list(iter_added_lines(patch)) == [(1, "new_call()")]


def test_iter_added_lines_handles_multiple_hunks():
    patch = (
        "@@ -1,1 +1,2 @@\n"
        " a = 1\n"
        "+b = 2\n"
        "@@ -10,1 +11,2 @@\n"
        " c = 3\n"
        "+d = 4\n"
    )
    assert list(iter_added_lines(patch)) == [(2, "b = 2"), (12, "d = 4")]


def test_iter_added_lines_ignores_diff_file_headers():
    patch = "--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,2 @@\n a = 1\n+b = 2\n"
    assert list(iter_added_lines(patch)) == [(2, "b = 2")]


def test_iter_added_lines_empty_patch_yields_nothing():
    assert list(iter_added_lines(None)) == []
    assert list(iter_added_lines("")) == []


def test_added_text_joins_only_added_lines():
    patch = "@@ -1,1 +1,3 @@\n context\n+first\n+second\n"
    assert added_text(patch) == "first\nsecond"


def test_numbered_added_lines_prefixes_real_line_numbers():
    patch = "@@ -1,1 +1,3 @@\n context\n+first\n+second\n"
    assert numbered_added_lines(patch) == "2: first\n3: second"


def test_numbered_added_lines_empty_patch_yields_empty_string():
    assert numbered_added_lines(None) == ""


def _finding(file_path="a.py", line_number=1, title="t") -> ReviewFinding:
    return ReviewFinding(
        file_path=file_path,
        line_number=line_number,
        title=title,
        severity=Severity.LOW,
        description="d",
        suggestion=None,
    )


def test_dedupe_findings_drops_repeat_of_file_line_title():
    first = _finding()
    duplicate = _finding()
    unique = _finding(line_number=2)
    assert dedupe_findings([first, duplicate, unique]) == [first, unique]


def test_dedupe_findings_keeps_first_occurrence():
    first = _finding(title="original")
    second = _finding(title="original")
    result = dedupe_findings([first, second])
    assert result == [first]


def test_no_newline_marker_does_not_shift_later_line_numbers():
    r"""git's "\ No newline at end of file" describes the preceding line; counting
    it as a line of its own pushed every later finding onto the wrong line."""
    patch = (
        "@@ -1,3 +1,4 @@\n"
        " ctx\n"
        "-old\n"
        "\\ No newline at end of file\n"
        "+new1\n"
    )
    assert list(iter_added_lines(patch)) == [(2, "new1")]


def test_added_line_starting_with_plus_is_kept_and_counted():
    """Inside a hunk a "+++" line is an added line whose text is "++", not a
    file header - dropping it silently desynced the numbering too."""
    patch = "@@ -1,1 +1,3 @@\n ctx\n+++x\n+after\n"
    assert list(iter_added_lines(patch)) == [(2, "++x"), (3, "after")]
