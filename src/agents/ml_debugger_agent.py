# src/agents/ml_debugger_agent.py
"""ML/DS anti-pattern specialist - the checks README.md describes as the
project's original purpose: vectorization failures, unbatched training,
missing GPU-inference context managers.

The regex checks below run unconditionally, same rationale as
security_agent.py. When GROQ_API_KEY is configured, a second pass hands
the diff to Groq (qwen/qwen3.8-27b) as a real tool-calling loop
(llm_client.call_llm_with_tools): the model can inspect a dataset's
schema, run statistical drift analysis, explain drift root cause via
SHAP, or check a documentation page for a deprecated API
(ml_debugger_tools.py) while it looks for ML/DS anti-patterns no fixed
pattern list can cover - train/test leakage, missing random seeds,
chained pandas assignment, unbatched inference, drifted fixtures, and
deprecated calls. That pass is additive and fails open.

The dataset tools only ever see repo-relative paths (whatever the model
read out of the diff), so - same reason drift_agent.py is built via a
factory closing over repo_path rather than being a bare function -
make_ml_debugger_agent(repo_path) binds them to a local PR checkout so
those paths resolve to real files instead of the server's cwd.
run_ml_debugger_agent is the repo_path=None instance, for direct/offline
use where no checkout is available (e.g. this module's own tests)."""

from __future__ import annotations

import re
from pathlib import Path

from src.agents.diff_utils import (
    added_text,
    dedupe_findings,
    iter_added_lines,
)
from src.agents.llm_client import call_llm_with_tools
from src.agents.llm_findings import collect_llm_findings
from src.agents.ml_debugger_tools import TOOL_SCHEMAS, make_tool_implementations
from src.schemas.graph import AgentNode, AgentState
from src.schemas.review import ReviewFinding, Severity

_ITERROWS = re.compile(r"\.iterrows\s*\(")
_EVAL_MODE = re.compile(r"\.eval\s*\(\s*\)")
_NO_GRAD = re.compile(r"no_grad\s*\(")
_FIT_CALL = re.compile(r"\.fit\s*\(")
_FOR_LOOP = re.compile(r"^\s*for\s+.+:\s*$")

_LLM_SYSTEM_PROMPT = """You are an ML/data-science code reviewer with tool access. You are
given the added lines of one file from a pull request diff, each prefixed
with its real line number in the new file. Find ML/DS-specific
anti-patterns that a fixed pattern list would miss - do NOT re-report
`.iterrows()`, a missing `torch.no_grad()` around `.eval()`, or `.fit()`
called inside a loop, those are already handled separately. Look instead
for things like: data leakage (fitting a scaler/encoder on the full
dataset before a train/test split, or fitting on test data), no random
seed set for a train_test_split/model that needs reproducibility, chained
pandas assignment (`df[df.x > 0]['y'] = ...`, which triggers
SettingWithCopyWarning and may silently no-op), computing a metric on a
GPU tensor without `.item()`/`.cpu()` first, and an unbounded/unbatched
DataFrame `.apply()` on a large table where a vectorized op would do.

You also have four tools:
- inspect_dataset_schema(file_path): a CSV/Parquet's shape, column dtypes,
  null counts, and numeric summary.
- run_drift_analysis(reference_path, current_path, columns=None):
  statistical drift between two versions of a dataset.
- explain_drift_root_cause(reference_path, current_path, top_n=5): SHAP
  attribution for which columns explain a shift already shown by
  run_drift_analysis - call this only after that, not instead of it.
- check_deprecated_api(symbol, doc_url): fetches a documentation/changelog
  URL and checks whether it flags `symbol` as deprecated.

Call a tool only when the diff gives you something concrete and real to
check - a .csv/.parquet path actually visible in the added lines, or a
library API call together with that library's real documentation URL.
Never invent a file path or URL that isn't grounded in what you were
given; if nothing warrants a tool call, just answer directly.

After any tool calls, respond with ONLY a JSON object of the shape:
{"findings": [{"line_number": <int|null>, "title": "<short title>",
"severity": "INFO"|"LOW"|"MEDIUM"|"HIGH"|"CRITICAL",
"description": "<why this is a problem>",
"suggestion": "<concrete fix>"}]}
If there are no additional issues, respond with {"findings": []}. Never
invent a line_number that wasn't given to you."""


def _first_matching_line(lines: list[tuple[int, str]], pattern: re.Pattern[str]) -> int | None:
    for line_no, text in lines:
        if pattern.search(text):
            return line_no
    return None


def _fit_call_inside_loop(lines: list[tuple[int, str]]) -> int | None:
    """Returns the line number of the first .fit() call that is indented
    deeper than a preceding for-loop line, i.e. actually nested inside it -
    not just present somewhere else in the same diff. Best-effort: it only
    sees added lines, so it can't detect nesting across an unchanged loop
    header, but it does rule out the common false positive of an unrelated
    loop and an unrelated top-level .fit() call both landing in one diff."""
    for i, (_, loop_text) in enumerate(lines):
        if not _FOR_LOOP.match(loop_text):
            continue
        loop_indent = len(loop_text) - len(loop_text.lstrip())
        for line_no, text in lines[i + 1 :]:
            if not text.strip():
                continue
            indent = len(text) - len(text.lstrip())
            if indent <= loop_indent:
                break
            if _FIT_CALL.search(text):
                return line_no
    return None


def make_ml_debugger_agent(repo_path: str | Path | None = None) -> AgentNode:
    """LangGraph node factory: PR changed_files -> ml_debugger_findings.

    repo_path anchors the Groq tool-calling pass's dataset tools
    (inspect_dataset_schema, run_drift_analysis, explain_drift_root_cause)
    to a real local checkout - without it those tools only ever see
    whatever the server process's cwd happens to contain, same failure
    mode drift_agent.py's factory exists to avoid."""
    tool_implementations = make_tool_implementations(repo_path)

    def run_ml_debugger_agent(state: AgentState) -> dict:
        pr = state["pr_payload"]
        findings: list[ReviewFinding] = []

        for changed_file in pr.changed_files:
            added = list(iter_added_lines(changed_file.patch))
            if not added:
                continue

            for line_no, text in added:
                if _ITERROWS.search(text):
                    findings.append(
                        ReviewFinding(
                            file_path=changed_file.filename,
                            line_number=line_no,
                            title="Row-wise DataFrame iteration via .iterrows()",
                            severity=Severity.HIGH,
                            description="iterrows() is orders of magnitude slower than a "
                            "vectorized operation because it boxes every row into a Series.",
                            suggestion="Replace with a vectorized column expression, or "
                            "Series.map()/apply() only if a true row-wise Python callback "
                            "is unavoidable.",
                        )
                    )

            block = added_text(changed_file.patch)
            if _EVAL_MODE.search(block) and not _NO_GRAD.search(block):
                findings.append(
                    ReviewFinding(
                        file_path=changed_file.filename,
                        line_number=_first_matching_line(added, _EVAL_MODE),
                        title="Inference call without torch.no_grad()",
                        severity=Severity.MEDIUM,
                        description="model.eval() switches off dropout/batchnorm training "
                        "behavior but does not stop autograd - without a no_grad() guard, "
                        "the forward pass still builds a graph and holds activations "
                        "it doesn't need for inference.",
                        suggestion="Wrap the inference calls in `with torch.no_grad():` "
                        "(or `torch.inference_mode()`).",
                    )
                )

            fit_in_loop_line = _fit_call_inside_loop(added)
            if fit_in_loop_line is not None:
                findings.append(
                    ReviewFinding(
                        file_path=changed_file.filename,
                        line_number=fit_in_loop_line,
                        title="Model .fit() called inside a loop",
                        severity=Severity.MEDIUM,
                        description="Calling .fit() repeatedly inside a Python loop "
                        "suggests unbatched, iterative training rather than a single "
                        "batched/epoch-based call.",
                        suggestion="Batch the inputs and call .fit() once per epoch (or "
                        "use a DataLoader with an explicit batch size) instead of looping "
                        "over individual examples.",
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

        return {"ml_debugger_findings": dedupe_findings(findings)}

    return run_ml_debugger_agent


# Default instance for direct/offline use (this module's own tests, and any
# caller with no PR checkout available) - dataset tool paths resolve as
# given, against the process's own cwd.
run_ml_debugger_agent = make_ml_debugger_agent()
