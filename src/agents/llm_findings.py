# src/agents/llm_findings.py
"""Shared LLM-findings plumbing for the specialist agents.

security_agent, architecture_agent and ml_debugger_agent each had a
verbatim copy of the same `_llm_findings` body: render the added lines,
ask the model, then build a ReviewFinding per item while tolerating
whatever the model actually returned. Only the *call* differs -
architecture uses the plain completion (`call_llm_findings`), security and
ml_debugger use the tool-calling loop (`call_llm_with_tools`) - so that
one difference is the `ask` argument and everything else lives here.

The per-file calls also run in a small thread pool. They are independent
network round trips, and a 20-file PR previously meant 20 sequential ones
on the critical path of a webhook job; the pool is capped because the
other side is a rate-limited API, not because of local CPU."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.agents.diff_utils import iter_added_lines, numbered_added_lines
from src.schemas.review import ReviewFinding, Severity

# ponytail: fixed cap, not a token-bucket. Raise it only alongside a real
# rate-limit strategy in llm_client.
_MAX_PARALLEL_FILES = 4

# `ask(system_prompt, user_prompt) -> raw findings, or None on any failure`
Ask = Callable[[str, str], list[dict[str, Any]] | None]


def parse_findings(
    filename: str,
    raw: Iterable[dict[str, Any]] | None,
    valid_lines: set[int] | None = None,
) -> list[ReviewFinding]:
    """Build ReviewFindings from the model's raw items, skipping any item
    that is missing a required field or carries a severity that isn't one
    of ours. An explicit `null` severity falls back to MEDIUM the same way
    a missing one does.

    The prompt tells the model never to invent a line_number outside what
    it was shown, but it's attacker-influenced text answering to a prompt,
    not a guarantee - and the model's tools (security_agent_tools.py,
    ml_debugger_tools.py) can themselves hand back whole-file line numbers
    that aren't positions in this diff. A line_number outside valid_lines
    is dropped to None (not the whole finding - it's still real content)
    rather than passed through: github_pr_comment.py anchors every
    non-None line_number as an inline comment, and GitHub's API rejects
    the *entire* review if even one comment lands outside the diff."""
    findings: list[ReviewFinding] = []
    for item in raw or []:
        try:
            line_number = item.get("line_number")
            if valid_lines is not None and line_number is not None and line_number not in valid_lines:
                line_number = None
            findings.append(
                ReviewFinding(
                    file_path=filename,
                    line_number=line_number,
                    title=item["title"],
                    severity=Severity((item.get("severity") or "MEDIUM").upper()),
                    description=item["description"],
                    suggestion=item.get("suggestion"),
                )
            )
        except (KeyError, ValueError, AttributeError, TypeError):
            continue
    return findings


def collect_llm_findings(changed_files: Sequence, system_prompt: str, ask: Ask) -> list[ReviewFinding]:
    """Run `ask` once per changed file that has added lines, and flatten the
    results in file order."""

    def for_file(changed_file) -> list[ReviewFinding]:
        numbered = numbered_added_lines(changed_file.patch)
        if not numbered:
            return []
        raw = ask(system_prompt, f"File: {changed_file.filename}\n\nAdded lines:\n{numbered}")
        valid_lines = {line_no for line_no, _ in iter_added_lines(changed_file.patch)}
        return parse_findings(changed_file.filename, raw, valid_lines)

    targets = [f for f in changed_files if f.patch]
    if not targets:
        return []
    if len(targets) == 1:
        return for_file(targets[0])

    with ThreadPoolExecutor(max_workers=min(_MAX_PARALLEL_FILES, len(targets))) as pool:
        # map preserves input order, so findings stay in file order
        return [finding for per_file in pool.map(for_file, targets) for finding in per_file]
