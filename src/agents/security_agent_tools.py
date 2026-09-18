# src/agents/security_agent_tools.py
"""Tool implementations + Groq tool-call schemas for security_agent's
function-calling pass - deterministic scans the model can call instead of
relying solely on its own judgment to spot secrets or unsafe code.

scan_secrets_regex is regex/entropy-based (re) - vendor-prefixed patterns
for the credential shapes that actually leak (AWS keys, OpenAI/Groq
tokens, GitHub/Slack tokens, PEM private key blocks, JWTs) plus a generic
high-entropy fallback for secret-like assignments that match no known
vendor prefix - with an optional detect-secrets CLI corroboration pass
when that tool is installed. run_ast_security_scan parses full Python
source with the stdlib `ast` module (needs syntactically valid code,
which a partial diff hunk usually isn't - hence it takes a file path, not
diff text) and flags unsafe calls by walking the tree, with an optional
bandit CLI corroboration pass when bandit is installed.

Same "deterministic core + optional corroborating tool" split as
ml_debugger_tools.run_drift_analysis's scipy+evidently pair: the regex/
AST pass is always the source of truth, detect-secrets/bandit are
additive and fail open (not installed, times out, crashes - all just
mean "no corroboration this run", never a hard failure of the tool call).

Every tool returns a JSON-serializable dict and never raises for an
"expected" failure - bad path, unparsable code, missing optional CLI -
the calling loop always needs something to feed back to the model."""

from __future__ import annotations

import ast
import json
import math
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.agents.repo_paths import resolve_repo_path

if TYPE_CHECKING:
    from groq.types.chat import ChatCompletionToolParam

_SUBPROCESS_TIMEOUT = 20.0

# Vendor-prefixed / structurally distinctive secret shapes - the same
# category of pattern detect-secrets ships as individual plugins
# (AWSKeyDetector, JwtTokenDetector, PrivateKeyDetector, ...), reimplemented
# directly so this check needs no optional dependency to run at all.
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("AWS access key ID", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "AWS secret access key",
        re.compile(r"(?i)aws_secret_access_key\s*[:=]\s*['\"][A-Za-z0-9/+=]{40}['\"]"),
    ),
    ("OpenAI API key", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("Groq API key", re.compile(r"\bgsk_[A-Za-z0-9]{20,}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")),
    (
        "Private key block",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ),
    (
        "JSON Web Token",
        re.compile(r"\bey[A-Za-z0-9_-]{10,}\.ey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    ),
)

# Catches a secret with no recognizable vendor prefix: a long, high-entropy
# literal assigned to a variable whose name says what it is.
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b\w*(api[_-]?key|secret|token|password|passwd|credential)\w*\s*[:=]\s*"
    r"['\"]([^'\"]{16,})['\"]"
)
_MIN_ENTROPY_BITS_PER_CHAR = 3.5


def _corroborate_with_cli(
    result: dict[str, Any],
    tool: str,
    args: list[str],
    report_key: str,
    note_key: str,
    absent_note: str,
) -> None:
    """Run an optional external scanner and fold its JSON into `result`.

    Both tools in this module follow the same contract, which is why it
    lives in one place: the project's own analysis is the source of truth,
    the CLI is corroboration, so a missing binary leaves a note rather than
    an error, the exit code is ignored (both exit non-zero simply for
    *finding* something), and any failure of the tool itself is caught -
    these run inside an LLM tool call that must always return something."""
    tool_path = shutil.which(tool)
    if tool_path is None:
        result[note_key] = absent_note
        return

    try:
        completed = subprocess.run(
            [tool_path, *args],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT,
        )
        result[report_key] = json.loads(completed.stdout) if completed.stdout else None
    except Exception as exc:  # the scanner's internals must never break the tool call
        result[note_key] = f"{tool} pass failed: {exc}"


def _shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return -sum((n / length) * math.log2(n / length) for n in counts.values())


def _high_entropy_assignment(line: str) -> str | None:
    match = _SECRET_ASSIGNMENT.search(line)
    if not match:
        return None
    if _shannon_entropy(match.group(2)) >= _MIN_ENTROPY_BITS_PER_CHAR:
        return "High-entropy value assigned to a secret-like name"
    return None


def _redact(line: str) -> str:
    """Never echo a found secret's actual value back into a tool result -
    that result becomes part of the LLM prompt/conversation history, which
    may be logged, so a real credential shouldn't flow through it either."""
    stripped = line.strip()
    if len(stripped) <= 12:
        return "*" * len(stripped)
    return f"{stripped[:4]}...{stripped[-4:]} ({len(stripped)} chars, redacted)"


def scan_secrets_regex(text: str) -> dict[str, Any]:
    """Regex/entropy scan over an arbitrary block of text (the diff's
    added lines, or a full file) for hardcoded secrets. Corroborated by
    the detect-secrets CLI when installed - not a hard dependency, this
    project's regex patterns are the source of truth either way."""
    matches: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for label, pattern in _SECRET_PATTERNS:
            if pattern.search(line):
                matches.append({"line": line_no, "kind": label, "snippet": _redact(line)})
        entropy_kind = _high_entropy_assignment(line)
        if entropy_kind:
            matches.append({"line": line_no, "kind": entropy_kind, "snippet": _redact(line)})

    result: dict[str, Any] = {"matches": matches, "n_matches": len(matches)}

    _corroborate_with_cli(
        result,
        "detect-secrets",
        ["scan", "--string", text],
        report_key="detect_secrets_report",
        note_key="detect_secrets_note",
        absent_note="detect-secrets CLI not installed; regex-only scan",
    )
    return result


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _is_attr_of(func: ast.expr, module: str) -> bool:
    return isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == module


def _is_dynamic_sql(arg: ast.expr) -> bool:
    """An f-string, %-format, or + concatenation as the first argument to
    execute()/executemany() - the shapes that build a SQL string instead
    of using parameter placeholders."""
    if isinstance(arg, ast.JoinedStr):
        return True
    if isinstance(arg, ast.BinOp) and isinstance(arg.op, (ast.Mod, ast.Add)):
        return True
    return False


class _UnsafeCallVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.findings: list[dict[str, Any]] = []

    def visit_Call(self, node: ast.Call) -> None:
        name = _call_name(node.func)

        if name in ("eval", "exec") and isinstance(node.func, ast.Name):
            self.findings.append(
                {
                    "line": node.lineno,
                    "kind": f"Use of {name}()",
                    "detail": f"{name}() executes its argument as Python code.",
                }
            )
        elif name in ("loads", "load", "Unpickler") and _is_attr_of(node.func, "pickle"):
            self.findings.append(
                {
                    "line": node.lineno,
                    "kind": "Untrusted deserialization via pickle",
                    "detail": "Unpickling can execute arbitrary code during deserialization.",
                }
            )
        elif name == "system" and _is_attr_of(node.func, "os"):
            self.findings.append(
                {
                    "line": node.lineno,
                    "kind": "Use of os.system()",
                    "detail": "Routes the command through the shell.",
                }
            )
        elif name in ("execute", "executemany") and node.args and _is_dynamic_sql(node.args[0]):
            self.findings.append(
                {
                    "line": node.lineno,
                    "kind": "Possible unparameterized SQL query",
                    "detail": "SQL built via string formatting/concatenation instead of "
                    "parameter placeholders (e.g. cursor.execute(query, params)).",
                }
            )

        for keyword in node.keywords:
            if (
                keyword.arg == "shell"
                and isinstance(keyword.value, ast.Constant)
                and keyword.value.value is True
            ):
                self.findings.append(
                    {
                        "line": node.lineno,
                        "kind": "subprocess call with shell=True",
                        "detail": "shell=True routes the command through the system shell.",
                    }
                )

        self.generic_visit(node)


def run_ast_security_scan(file_path: str) -> dict[str, Any]:
    """Parses a full Python file's AST and flags unsafe calls. Needs real,
    syntactically valid source - a diff hunk of just added lines usually
    isn't one, which is why this takes a file path rather than diff text.
    Corroborated by the bandit CLI when installed."""
    try:
        source = Path(file_path).read_text(encoding="utf-8")
    except OSError as exc:
        return {"error": str(exc)}

    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError as exc:
        return {"error": f"could not parse {file_path} as Python: {exc}"}

    visitor = _UnsafeCallVisitor()
    visitor.visit(tree)
    result: dict[str, Any] = {
        "file_path": file_path,
        "findings": visitor.findings,
        "n_findings": len(visitor.findings),
    }

    _corroborate_with_cli(
        result,
        "bandit",
        ["-f", "json", "-q", file_path],
        report_key="bandit_report",
        note_key="bandit_note",
        absent_note="bandit CLI not installed; ast-only scan",
    )
    return result


TOOL_SCHEMAS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "scan_secrets_regex",
            "description": (
                "Scan a block of text/code for hardcoded secrets - AWS access keys, "
                "OpenAI/Groq API tokens, GitHub/Slack tokens, PEM private key blocks, "
                "JWTs, and high-entropy values assigned to secret-like variable names. "
                "Corroborated by the detect-secrets CLI when installed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The code/text to scan, e.g. the diff's added lines.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_ast_security_scan",
            "description": (
                "Parse a Python file's full AST and flag unsafe calls: eval()/exec(), "
                "pickle.loads/Unpickler, os.system(), subprocess(shell=True), and SQL "
                "built via string formatting/f-strings/concatenation passed to "
                "execute()/executemany(). Needs a real file path - a diff hunk alone is "
                "usually not valid, parseable Python. Corroborated by bandit when "
                "installed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to a Python file visible in the diff/repo.",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
]

TOOL_IMPLEMENTATIONS: dict[str, Any] = {
    "scan_secrets_regex": scan_secrets_regex,
    "run_ast_security_scan": run_ast_security_scan,
}


def make_tool_implementations(repo_path: str | Path | None = None) -> dict[str, Any]:
    """Binds run_ast_security_scan to a local PR checkout so relative file
    paths the model supplies resolve against real repo content instead of
    the server's cwd. scan_secrets_regex takes text directly, not a path,
    so it passes through unchanged. Mirrors
    ml_debugger_tools.make_tool_implementations / drift_agent.make_drift_agent's
    repo_path factory pattern."""
    if repo_path is None:
        return dict(TOOL_IMPLEMENTATIONS)

    def _run_ast_security_scan(file_path: str) -> dict[str, Any]:
        return run_ast_security_scan(resolve_repo_path(repo_path, file_path))

    return {
        "scan_secrets_regex": scan_secrets_regex,
        "run_ast_security_scan": _run_ast_security_scan,
    }
