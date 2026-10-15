# src/agents/ml_debugger_tools.py
"""Tool implementations + Groq tool-call schemas for ml_debugger_agent's
function-calling pass.

These are agent-scoped, not orchestrator-scoped: src/tools/tools.py wraps
EvidenceRetriever for orchestrator.diagnose()'s fixed plan -> gather ->
hypothesize pipeline, but ml_debugger_agent's Groq pass is a genuine
tool-calling loop where the model itself decides which file paths/URLs are
worth inspecting, so these tools take plain LLM-suppliable arguments
(file paths, a symbol name, a doc URL) instead of an injected retriever.

Every tool returns a JSON-serializable dict and never raises for an
"expected" failure (bad path, unreadable file, network error) - the
calling loop always needs something to feed back to the model as the tool
result, the same fail-open contract as llm_client.call_llm_findings. Only
a genuinely unexpected exception should escape, and llm_client's tool-call
loop catches those too before they can take down the whole review."""

from __future__ import annotations

import ipaddress
import re
import socket
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx

from src.agents.repo_paths import resolve_repo_path
from src.drift.detector import detect_dataset_drift, format_report
from src.drift.methods import DriftInputError
from src.drift.root_cause import RootCauseInputError, explain_drift, format_root_cause
from src.drift.tables import read_table

if TYPE_CHECKING:
    # Type-checking only, same reason llm_client.py guards its groq
    # TypedDict imports the same way - groq stays an optional dependency.
    from groq.types.chat import ChatCompletionToolParam

_MAX_DOC_CHARS = 200_000
_DOC_FETCH_TIMEOUT = 10.0
_SNIPPET_RADIUS = 200
_MAX_SNIPPETS = 5
_MAX_REDIRECTS = 5


def inspect_dataset_schema(file_path: str) -> dict[str, Any]:
    """pandas-based schema/profile: shape, per-column dtype/null/unique
    counts, and numeric summary stats - lets the model see what a dataset
    actually looks like before deciding a drift or root-cause check is
    worth running on it."""
    try:
        frame = read_table(file_path)
    except (OSError, ValueError) as exc:
        return {"error": str(exc)}

    numeric = frame.select_dtypes(include="number")
    describe = numeric.describe().to_dict() if not numeric.empty else {}
    return {
        "file_path": file_path,
        "n_rows": int(frame.shape[0]),
        "n_columns": int(frame.shape[1]),
        "columns": {
            str(column): {
                "dtype": str(frame[column].dtype),
                "n_null": int(frame[column].isna().sum()),
                "n_unique": int(frame[column].nunique(dropna=True)),
            }
            for column in frame.columns
        },
        "numeric_summary": {
            str(column): {stat: float(value) for stat, value in stats.items()}
            for column, stats in describe.items()
        },
    }


def run_drift_analysis(
    reference_path: str, current_path: str, columns: list[str] | None = None
) -> dict[str, Any]:
    """scipy-backed drift verdict (src/drift/detector.py - the same engine
    drift_agent.py uses for the PR-review path) plus an Evidently
    DataDriftPreset pass as a second, corroborating signal when the
    optional `evidently` package is installed. scipy stays the source of
    truth for `drifted`; Evidently is corroboration/detail only, the same
    division of labor as demo/evidently_dashboard.py - so this tool works
    identically whether or not evidently is installed."""
    try:
        reference = read_table(reference_path)
        current = read_table(current_path)
    except (OSError, ValueError) as exc:
        return {"error": str(exc)}

    try:
        report = detect_dataset_drift(
            reference, current, columns=columns, label=f"{reference_path} -> {current_path}"
        )
    except DriftInputError as exc:
        return {"error": str(exc)}

    result: dict[str, Any] = {
        "scipy_report": report.model_dump(mode="json"),
        "summary": format_report(report),
    }

    try:
        from evidently import Dataset, Report
        from evidently.presets import DataDriftPreset

        # reuse the columns detect_dataset_drift actually compared, rather
        # than re-deriving them from the raw `columns` argument
        shared = [c.column for c in report.columns]
        evidently_report = Report(metrics=[DataDriftPreset()]).run(
            reference_data=Dataset.from_pandas(reference[shared]),
            current_data=Dataset.from_pandas(current[shared]),
        )
        result["evidently_summary"] = evidently_report.dict()
    except ImportError:
        result["evidently_summary"] = None
        result["evidently_note"] = "evidently not installed; scipy verdict only"
    except Exception as exc:  # evidently internals must never break the tool call
        result["evidently_summary"] = None
        result["evidently_note"] = f"evidently pass failed: {exc}"

    return result


def explain_drift_root_cause(
    reference_path: str, current_path: str, top_n: int = 5
) -> dict[str, Any]:
    """SHAP (TreeExplainer over an XGBoost surrogate) attribution for which
    columns actually explain a reference -> current shift - a thin wrapper
    around drift/root_cause.explain_drift. Only meaningful after
    run_drift_analysis has shown the two datasets are comparable and
    plausibly different; the report's own `separable` gate still needs to
    be checked before its ranking means anything."""
    try:
        reference = read_table(reference_path)
        current = read_table(current_path)
    except (OSError, ValueError) as exc:
        return {"error": str(exc)}

    try:
        report = explain_drift(reference, current, label=f"{reference_path} -> {current_path}")
    except RootCauseInputError as exc:
        return {"error": str(exc)}

    return {
        "report": report.model_dump(mode="json"),
        "summary": format_root_cause(report, top_n=top_n),
    }


_DEPRECATION_KEYWORDS = re.compile(
    r"deprecat|removed in|will be removed|use .*? instead|superseded by", re.IGNORECASE
)


def _strip_html(html: str) -> str:
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _unsafe_url_reason(url: str) -> str | None:
    """None if url is safe to fetch, else why it isn't.

    doc_url comes from the model, whose only input is the PR diff, so a PR
    can simply ask for `http://169.254.169.254/latest/meta-data/...` and get
    the worker to fetch it from inside the network and quote the response
    back. https-only plus a public-address check is the floor.

    ponytail: resolve-then-connect leaves a DNS-rebinding window; pin the
    resolved address into the connection if that ever matters here."""
    parsed = urlparse(url)
    if parsed.scheme != "https":
        return f"only https doc URLs are allowed, got {parsed.scheme or 'no'} scheme"
    if not parsed.hostname:
        return "no hostname in URL"
    try:
        infos = socket.getaddrinfo(parsed.hostname, parsed.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        return f"could not resolve {parsed.hostname}: {exc}"
    for info in infos:
        address = ipaddress.ip_address(info[4][0])
        if not address.is_global:
            return f"{parsed.hostname} resolves to non-public address {address}"
    return None


def check_deprecated_api(symbol: str, doc_url: str) -> dict[str, Any]:
    """Fetches the documentation/changelog page the model names and checks
    whether it mentions `symbol` alongside deprecation language.

    This is a live lookup against whatever the model believes are the
    real, current docs for the library in question, rather than a
    hand-maintained table of known-deprecated APIs baked into this
    project - the model supplies the URL because it (not this codebase)
    knows which library and version the diff is using. That makes the URL
    untrusted input, so every hop is screened by _unsafe_url_reason."""
    # Redirects are followed by hand so every hop is re-checked - a single
    # allowed URL that 302s to 127.0.0.1 would otherwise walk straight past
    # the guard.
    url = doc_url
    with httpx.Client(
        timeout=_DOC_FETCH_TIMEOUT,
        follow_redirects=False,
        headers={"User-Agent": "skynet-ds-ml-debugger/1.0"},
    ) as client:
        for _ in range(_MAX_REDIRECTS):
            reason = _unsafe_url_reason(url)
            if reason:
                return {"error": f"refusing to fetch {url}: {reason}"}
            try:
                response = client.get(url)
            except httpx.HTTPError as exc:
                return {"error": f"could not fetch {url}: {exc}"}
            if response.is_redirect and response.next_request is not None:
                url = str(response.next_request.url)
                continue
            break
        else:
            return {"error": f"too many redirects from {doc_url}"}

    try:
        response.raise_for_status()
    except httpx.HTTPError as exc:
        return {"error": f"could not fetch {url}: {exc}"}

    text = _strip_html(response.text[:_MAX_DOC_CHARS])
    snippets: list[str] = []
    for match in re.finditer(re.escape(symbol), text, re.IGNORECASE):
        window = text[max(0, match.start() - _SNIPPET_RADIUS) : match.start() + _SNIPPET_RADIUS]
        if _DEPRECATION_KEYWORDS.search(window):
            snippets.append(window.strip())
        if len(snippets) >= _MAX_SNIPPETS:
            break

    return {
        "symbol": symbol,
        "doc_url": doc_url,
        "likely_deprecated": bool(snippets),
        "evidence_snippets": snippets,
    }


TOOL_SCHEMAS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "inspect_dataset_schema",
            "description": (
                "Read a CSV/Parquet file with pandas and return its shape, column "
                "dtypes, null counts, and numeric summary stats."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to a .csv or .parquet file visible in the diff/repo.",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_drift_analysis",
            "description": (
                "Compare a reference and current CSV/Parquet file for statistical data "
                "drift (scipy.stats KS/PSI/etc., corroborated by Evidently's "
                "DataDriftPreset when available). Use once you have two real, comparable "
                "dataset paths, e.g. from inspect_dataset_schema."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reference_path": {"type": "string", "description": "Baseline/reference dataset path."},
                    "current_path": {
                        "type": "string",
                        "description": "Current/production dataset path to compare against the reference.",
                    },
                    "columns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional subset of shared columns to check; defaults to all shared columns.",
                    },
                },
                "required": ["reference_path", "current_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_drift_root_cause",
            "description": (
                "Given two datasets that run_drift_analysis showed as drifted, train a "
                "small surrogate classifier and use SHAP (TreeExplainer) to rank which "
                "columns actually explain the shift. Only call this after "
                "run_drift_analysis, not as a first step."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reference_path": {"type": "string"},
                    "current_path": {"type": "string"},
                    "top_n": {
                        "type": "integer",
                        "description": "How many top features to return, default 5.",
                    },
                },
                "required": ["reference_path", "current_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_deprecated_api",
            "description": (
                "Fetch a documentation/changelog/API-reference URL and check whether it "
                "flags the given function/API name as deprecated. Use when the diff calls "
                "a library function you suspect is deprecated and you know that library's "
                "real docs URL - never guess a URL you are not reasonably confident is "
                "real."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "The function or API name to check, e.g. 'pandas.DataFrame.append'.",
                    },
                    "doc_url": {
                        "type": "string",
                        "description": "URL of the documentation/changelog page to check against.",
                    },
                },
                "required": ["symbol", "doc_url"],
            },
        },
    },
]

TOOL_IMPLEMENTATIONS: dict[str, Any] = {
    "inspect_dataset_schema": inspect_dataset_schema,
    "run_drift_analysis": run_drift_analysis,
    "explain_drift_root_cause": explain_drift_root_cause,
    "check_deprecated_api": check_deprecated_api,
}


def make_tool_implementations(repo_path: str | Path | None = None) -> dict[str, Any]:
    """Binds the dataset-reading tools to a local PR checkout so relative
    file paths the model supplies resolve against real repo content
    instead of the server's cwd. check_deprecated_api takes a URL, not a
    repo path, so it passes through unchanged. Mirrors
    drift_agent.make_drift_agent's repo_path factory pattern.

    Without a checkout there is no repo tree to confine a path to -
    resolve_repo_path(None, ...) is a no-op (see its docstring), so the
    three file-reading tools would open any path the model names, anywhere
    on the server. Drop them rather than hand them out unconfined;
    check_deprecated_api has its own SSRF guard and isn't path-based, so
    it stays."""
    if repo_path is None:
        return {"check_deprecated_api": check_deprecated_api}

    def _inspect_dataset_schema(file_path: str) -> dict[str, Any]:
        return inspect_dataset_schema(resolve_repo_path(repo_path, file_path))

    def _run_drift_analysis(
        reference_path: str, current_path: str, columns: list[str] | None = None
    ) -> dict[str, Any]:
        return run_drift_analysis(
            resolve_repo_path(repo_path, reference_path),
            resolve_repo_path(repo_path, current_path),
            columns,
        )

    def _explain_drift_root_cause(
        reference_path: str, current_path: str, top_n: int = 5
    ) -> dict[str, Any]:
        return explain_drift_root_cause(
            resolve_repo_path(repo_path, reference_path),
            resolve_repo_path(repo_path, current_path),
            top_n,
        )

    return {
        "inspect_dataset_schema": _inspect_dataset_schema,
        "run_drift_analysis": _run_drift_analysis,
        "explain_drift_root_cause": _explain_drift_root_cause,
        "check_deprecated_api": check_deprecated_api,
    }
