# src/agents/security_agent.py
"""Security specialist: regex checks over a PR's added lines for the
vulnerability classes most likely to slip into an ML/DS codebase (unsafe
deserialization, shell injection, disabled TLS verification, hardcoded
credentials).

The regex checks below run unconditionally - they're free, instant, and
what the test suite exercises. When GROQ_API_KEY is configured
(src/core/config.py), a second pass hands the diff to Groq (qwen/qwen3-32b,
src/agents/llm_client.py) as a real tool-calling loop: instead of relying
solely on the model's own judgment, it can call scan_secrets_regex (regex/
entropy secret patterns, optionally corroborated by detect-secrets) and
run_ast_security_scan (stdlib `ast`-based unsafe-call detection, optionally
corroborated by bandit) - security_agent_tools.py - while it looks for
vulnerability classes no fixed pattern list can cover: SSRF, path
traversal, insecure randomness for security-sensitive values, and so on.
The LLM pass is additive and fails open: no key, an SDK error, or
unparseable output all just mean "no extra findings this run".

run_ast_security_scan needs a real file path to parse (a diff hunk of
just added lines usually isn't valid, parseable Python on its own), so -
same reason drift_agent.py and ml_debugger_agent.py are built via a
factory closing over repo_path rather than being a bare function -
make_security_agent(repo_path) binds it to a local PR checkout.
run_security_agent is the repo_path=None instance, for direct/offline use
where no checkout is available (e.g. this module's own tests)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.agents.diff_utils import dedupe_findings, iter_added_lines
from src.agents.llm_client import call_llm_with_tools
from src.agents.llm_findings import collect_llm_findings
from src.agents.security_agent_tools import TOOL_SCHEMAS, make_tool_implementations
from src.schemas.graph import AgentNode, AgentState
from src.schemas.review import ReviewFinding, Severity


@dataclass(frozen=True)
class _Check:
    pattern: re.Pattern[str]
    title: str
    severity: Severity
    description: str
    suggestion: str


_CHECKS: tuple[_Check, ...] = (
    _Check(
        re.compile(r"(?<![\w.])eval\s*\("),
        "Use of eval()",
        Severity.HIGH,
        "eval() executes its argument as Python code; if any part of that "
        "string can be influenced by untrusted input, this is a code-injection vector.",
        "Replace eval() with ast.literal_eval() for literals, or an explicit "
        "parser/dispatch table for anything more complex.",
    ),
    _Check(
        re.compile(r"(?<![\w.])exec\s*\("),
        "Use of exec()",
        Severity.HIGH,
        "exec() runs arbitrary Python source at runtime; treat it the same "
        "as eval() for injection risk.",
        "Avoid exec() - restructure the code so behavior is chosen via "
        "normal function calls or a dispatch table instead of generated code.",
    ),
    _Check(
        re.compile(r"pickle\.(loads?|Unpickler)\s*\("),
        "Untrusted deserialization via pickle",
        Severity.HIGH,
        "Unpickling can execute arbitrary code during deserialization - "
        "never unpickle data from a source you don't fully control (a "
        "user upload, a network response, a shared cache).",
        "Use a safe serialization format for cross-boundary data (JSON, "
        "safetensors, numpy .npy) or verify the source is fully trusted.",
    ),
    _Check(
        re.compile(r"yaml\.load\s*\((?![^)]*Loader\s*=\s*yaml\.SafeLoader)"),
        "yaml.load() without SafeLoader",
        Severity.MEDIUM,
        "yaml.load() without an explicit safe Loader can construct "
        "arbitrary Python objects from the YAML content.",
        "Use yaml.safe_load(), or pass Loader=yaml.SafeLoader explicitly.",
    ),
    _Check(
        re.compile(r"subprocess\.\w+\([^)]*shell\s*=\s*True"),
        "subprocess call with shell=True",
        Severity.HIGH,
        "shell=True routes the command through the system shell, so any "
        "unsanitized input concatenated into it becomes a shell-injection vector.",
        "Pass the command as a list of arguments and drop shell=True.",
    ),
    _Check(
        re.compile(r"os\.system\s*\("),
        "Use of os.system()",
        Severity.MEDIUM,
        "os.system() runs its argument through the shell, with the same "
        "injection risk as subprocess(..., shell=True).",
        "Use subprocess.run([...]) with an argument list instead.",
    ),
    _Check(
        re.compile(r"verify\s*=\s*False"),
        "TLS certificate verification disabled",
        Severity.MEDIUM,
        "verify=False accepts any certificate, exposing the connection to "
        "man-in-the-middle interception.",
        "Remove verify=False and fix the underlying certificate problem instead.",
    ),
    _Check(
        re.compile(
            r"(?i)\b(api[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*['\"][^'\"]{4,}['\"]"
        ),
        "Hardcoded credential",
        Severity.CRITICAL,
        "A literal secret is checked directly into source control, where "
        "it stays in git history even if removed later.",
        "Move the value to an environment variable or secrets manager and "
        "load it at runtime (see src/core/config.py's pattern).",
    ),
)


_LLM_SYSTEM_PROMPT = """You are a security reviewer for a Python data-science/ML codebase with
tool access. You are given the added lines of one file from a pull
request diff, each prefixed with its real line number in the new file.
Find security vulnerabilities that a fixed regex list would miss - do NOT
re-report eval()/exec()/pickle/yaml.load/shell=True/os.system/verify=False/
hardcoded credentials, those are already handled separately. Look instead
for things like: SSRF (fetching a URL built from unsanitized input), path
traversal (joining a path from unsanitized input), insecure randomness
used for a security-sensitive value (tokens, secrets) via `random` instead
of `secrets`, XML parsing without disabling external entities, insecure
temp file handling, and overly permissive file/network permissions.

You also have two tools:
- scan_secrets_regex(text): scans a block of text/code for hardcoded
  secrets (AWS/OpenAI/Groq/GitHub/Slack tokens, private key blocks, JWTs,
  high-entropy secret-like assignments). Use it on the added lines you
  were given if anything looks like it could be a credential.
- run_ast_security_scan(file_path): parses a full Python file's AST for
  unsafe calls (eval/exec, pickle deserialization, os.system,
  subprocess(shell=True), unparameterized SQL). Needs a real file path -
  use the file path you were given, not the diff snippet itself.

Only call a tool when it can add something real; if nothing warrants it,
just answer directly.

Respond with ONLY a JSON object of the shape:
{"findings": [{"line_number": <int>, "title": "<short title>",
"severity": "INFO"|"LOW"|"MEDIUM"|"HIGH"|"CRITICAL",
"description": "<why this is a problem>",
"suggestion": "<concrete fix>"}]}
If there are no additional issues, respond with {"findings": []}. Never
invent a line_number that wasn't given to you."""


def make_security_agent(repo_path: str | Path | None = None) -> AgentNode:
    """LangGraph node factory: PR changed_files -> security_findings.

    repo_path anchors the Groq tool-calling pass's run_ast_security_scan
    to a real local checkout, so the file path it's given resolves to
    real source instead of whatever the server process's cwd contains -
    same failure mode drift_agent.py's and ml_debugger_agent.py's
    factories exist to avoid."""
    tool_implementations = make_tool_implementations(repo_path)

    def run_security_agent(state: AgentState) -> dict:
        pr = state["pr_payload"]
        findings: list[ReviewFinding] = []

        for changed_file in pr.changed_files:
            for line_no, text in iter_added_lines(changed_file.patch):
                for check in _CHECKS:
                    if check.pattern.search(text):
                        findings.append(
                            ReviewFinding(
                                file_path=changed_file.filename,
                                line_number=line_no,
                                title=check.title,
                                severity=check.severity,
                                description=check.description,
                                suggestion=check.suggestion,
                            )
                        )
        findings.extend(
            collect_llm_findings(
                pr.changed_files,
                _LLM_SYSTEM_PROMPT,
                lambda system, user: call_llm_with_tools(
                    system, user, TOOL_SCHEMAS, tool_implementations
                ),
            )
        )

        return {"security_findings": dedupe_findings(findings)}

    return run_security_agent


# Default instance for direct/offline use (this module's own tests, and any
# caller with no PR checkout available) - run_ast_security_scan's file
# path resolves as given, against the process's own cwd.
run_security_agent = make_security_agent()
