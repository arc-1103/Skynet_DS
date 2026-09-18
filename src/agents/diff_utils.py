# src/agents/diff_utils.py
"""Shared unified-diff parsing for the specialist agents in this package.

Every specialist agent (security, architecture, ml_debugger) works off
`ChangedFile.patch` - a unified diff hunk, not a full file - so a finding
can only ever be anchored to an *added* line, and only the added side of
the diff is worth scanning: a pattern already present on the base branch
isn't something this PR introduced."""

from __future__ import annotations

import re
from collections.abc import Iterator

_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<new_start>\d+)(?:,\d+)? @@")


def iter_added_lines(patch: str | None) -> Iterator[tuple[int, str]]:
    """Yield (line_number_in_new_file, line_text) for each added line.

    line_number_in_new_file is the line's position in the post-PR version of
    the file, matching what GitHub's inline-comment API expects. Lines
    outside any hunk (the ---/+++ file headers, and anything else preceding
    the first @@) are ignored rather than mis-numbered, and git's
    "no newline at end of file" marker is not counted as a line."""
    if not patch:
        return

    new_line_no: int | None = None
    for line in patch.splitlines():
        header = _HUNK_HEADER.match(line)
        if header:
            new_line_no = int(header.group("new_start"))
            continue
        if new_line_no is None:
            continue

        if line.startswith("+"):
            yield new_line_no, line[1:]
            new_line_no += 1
        elif line.startswith("-"):
            continue
        elif line.startswith("\\"):
            # "\ No newline at end of file" - a marker about the preceding
            # line, not a line of its own. Counting it shifted every
            # subsequent finding's line number by one.
            continue
        else:
            new_line_no += 1


def added_text(patch: str | None) -> str:
    """All added lines joined back into one block, for checks that need to
    know whether a pattern appears anywhere in the file's added code rather
    than on a specific line (e.g. "is `no_grad` mentioned at all")."""
    return "\n".join(text for _, text in iter_added_lines(patch))


def numbered_added_lines(patch: str | None) -> str:
    """Added lines rendered as "<line_no>: <text>", one per line - the form
    handed to the LLM-backed analysis pass (llm_client.py) so the model can
    cite a real line_number directly instead of guessing an offset into the
    raw patch."""
    return "\n".join(f"{line_no}: {text}" for line_no, text in iter_added_lines(patch))


def dedupe_findings(findings: list) -> list:
    """Drops findings that repeat an earlier one's (file_path, line_number,
    title) - used to merge a specialist's deterministic regex findings with
    its LLM-backed findings without showing the same issue twice when both
    passes catch it. First occurrence wins, so regex findings (added first)
    take priority over an LLM finding describing the same thing."""
    seen: set[tuple[str, int | None, str]] = set()
    deduped = []
    for finding in findings:
        key = (finding.file_path, finding.line_number, finding.title)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return deduped
