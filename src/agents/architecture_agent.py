# src/agents/architecture_agent.py
"""Architecture/code-quality specialist: structural smells detectable from
a diff hunk alone (no full-file fetch needed) - bare excepts, print-based
logging, deep nesting, and diffs too large to review carefully.

The regex checks below run unconditionally, same rationale as
security_agent.py. When GROQ_API_KEY is configured, a second pass hands
the diff to Groq (qwen/qwen3.8-27b) for structural issues no fixed pattern
list can cover - god functions, duplicated logic, unclear naming, tight
coupling, missing type hints. That pass is additive and fails open."""

from __future__ import annotations

import re

from src.agents.diff_utils import dedupe_findings, iter_added_lines
from src.agents.llm_client import call_llm_findings
from src.agents.llm_findings import collect_llm_findings
from src.schemas.graph import AgentState
from src.schemas.review import ReviewFinding, Severity

_BARE_EXCEPT = re.compile(r"^\s*except\s*:\s*$")
_PRINT_CALL = re.compile(r"^\s*print\s*\(")
_LARGE_DIFF_THRESHOLD = 300
_DEEP_NESTING_SPACES = 16  # 4 levels of 4-space indentation

_LLM_SYSTEM_PROMPT = """You are a code-quality/architecture reviewer for a Python
data-science/ML codebase. You are given the added lines of one file from a
pull request diff, each prefixed with its real line number in the new
file. Find structural/architecture issues that a fixed pattern list would
miss - do NOT re-report bare `except:`, `print()` instead of logging, or
generically "deeply nested code", those are already handled separately.
Look instead for things like: a function doing too many unrelated things
(god function), duplicated logic that should be extracted into a shared
helper, unclear/misleading naming, tight coupling to a concrete
implementation where an interface would do, missing type hints on a
public function signature, and mutable default arguments.

Respond with ONLY a JSON object of the shape:
{"findings": [{"line_number": <int|null>, "title": "<short title>",
"severity": "INFO"|"LOW"|"MEDIUM"|"HIGH"|"CRITICAL",
"description": "<why this is a problem>",
"suggestion": "<concrete fix>"}]}
If there are no additional issues, respond with {"findings": []}. Never
invent a line_number that wasn't given to you."""


def run_architecture_agent(state: AgentState) -> dict:
    """LangGraph node: PR changed_files -> architecture_findings."""
    pr = state["pr_payload"]
    findings: list[ReviewFinding] = []

    for changed_file in pr.changed_files:
        if changed_file.additions > _LARGE_DIFF_THRESHOLD:
            findings.append(
                ReviewFinding(
                    file_path=changed_file.filename,
                    line_number=None,
                    title="Large diff in a single file",
                    severity=Severity.MEDIUM,
                    description=f"This file has {changed_file.additions} added lines "
                    "in one PR, which is hard to review carefully and increases the "
                    "chance a regression slips through.",
                    suggestion="Split this into smaller, focused PRs or commits where possible.",
                )
            )

        for line_no, text in iter_added_lines(changed_file.patch):
            if _BARE_EXCEPT.match(text):
                findings.append(
                    ReviewFinding(
                        file_path=changed_file.filename,
                        line_number=line_no,
                        title="Bare except clause",
                        severity=Severity.MEDIUM,
                        description="except: with no exception type silently swallows "
                        "every error, including KeyboardInterrupt and SystemExit, "
                        "making failures invisible.",
                        suggestion="Catch a specific exception type (or at least "
                        "Exception) and log or re-raise it.",
                    )
                )
            elif _PRINT_CALL.match(text):
                findings.append(
                    ReviewFinding(
                        file_path=changed_file.filename,
                        line_number=line_no,
                        title="print() used instead of logging",
                        severity=Severity.LOW,
                        description="print() output isn't leveled or capturable the "
                        "way this codebase's structlog/logging setup is.",
                        suggestion="Use the logging module (e.g. logger.info(...)) instead of print().",
                    )
                )
            elif len(text) - len(text.lstrip(" ")) >= _DEEP_NESTING_SPACES and text.strip():
                findings.append(
                    ReviewFinding(
                        file_path=changed_file.filename,
                        line_number=line_no,
                        title="Deeply nested logic",
                        severity=Severity.LOW,
                        description="This line is indented 4+ levels deep, which "
                        "usually means the enclosing function is doing too much and "
                        "is hard to follow or test in isolation.",
                        suggestion="Extract the nested block into a well-named helper function.",
                    )
                )

    findings.extend(
        collect_llm_findings(pr.changed_files, _LLM_SYSTEM_PROMPT, call_llm_findings)
    )

    return {"architecture_findings": dedupe_findings(findings)}
