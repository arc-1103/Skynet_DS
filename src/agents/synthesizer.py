# src/agents/synthesizer.py
"""Fan-in node: merges security_findings, architecture_findings,
ml_debugger_findings, and drift_report into one FinalPRReview.

Runs after all four specialist branches complete (LangGraph only schedules
a node once every edge into it has fired), so this is the one place that
sees the whole picture and decides pass/fail and labels."""

from __future__ import annotations

from src.schemas.graph import AgentState
from src.schemas.review import FinalPRReview, PRLabel, ReviewFinding, Severity

_BLOCKING_SEVERITIES = frozenset({Severity.HIGH, Severity.CRITICAL})


def _labels_for(
    security: list[ReviewFinding],
    architecture: list[ReviewFinding],
    ml_debugger: list[ReviewFinding],
    drift_flagged: bool,
) -> list[PRLabel]:
    labels: list[PRLabel] = []
    if security:
        labels.append(PRLabel.SECURITY)
    if ml_debugger:
        labels.append(PRLabel.PERFORMANCE)
    if architecture:
        labels.append(PRLabel.REFACTOR)
    if drift_flagged:
        labels.append(PRLabel.CONFIG)
    return labels


def _drop_already_reported(findings: list[ReviewFinding], *earlier: list[ReviewFinding]) -> list[ReviewFinding]:
    """Titles differ across agents (Security's "Use of eval()" vs
    Architecture's LLM "Use of eval()" at a different severity), so the
    per-agent title dedupe can't catch these. A higher-priority agent that
    already flagged a (file, line) owns it; file-level findings (no line)
    are never dropped."""
    taken = {(f.file_path, f.line_number) for group in earlier for f in group if f.line_number is not None}
    return [f for f in findings if f.line_number is None or (f.file_path, f.line_number) not in taken]


def _section(title: str, findings: list[ReviewFinding]) -> str:
    if not findings:
        return f"### {title}\nNo issues found.\n"
    lines = [f"### {title} ({len(findings)})"]
    for f in findings:
        location = f"{f.file_path}:{f.line_number}" if f.line_number is not None else f.file_path
        lines.append(f"- **[{f.severity.value}]** `{location}` — {f.title}: {f.description}")
    return "\n".join(lines) + "\n"


def _drift_section(reports) -> tuple[str, bool]:
    if not reports:
        return "### Data Drift\nNo tabular data files changed in this PR.\n", False
    flagged = [r for r in reports if r.dataset_drifted]
    lines = [f"### Data Drift ({len(flagged)}/{len(reports)} file(s) flagged)"]
    for r in reports:
        verdict = "DRIFTED" if r.dataset_drifted else "stable"
        lines.append(
            f"- `{r.label}`: {verdict} ({r.n_drifted}/{r.n_columns} columns, "
            f"share={r.share_drifted:.2f})"
        )
    return "\n".join(lines) + "\n", bool(flagged)


def run_synthesizer_agent(state: AgentState) -> dict:
    """LangGraph node: all specialist outputs -> final_review."""
    security = state.get("security_findings", [])
    ml_debugger = _drop_already_reported(state.get("ml_debugger_findings", []), security)
    architecture = _drop_already_reported(state.get("architecture_findings", []), security, ml_debugger)
    drift_reports = state.get("drift_report", [])

    drift_text, drift_flagged = _drift_section(drift_reports)

    inline_findings = [*security, *architecture, *ml_debugger]
    # drift_flagged counts too: a PR whose data files all drifted already
    # gets a CONFIG label from _labels_for, and rendering "PASSED" next to
    # it contradicts the verdict this node exists to produce.
    overall_passed = not drift_flagged and not any(
        f.severity in _BLOCKING_SEVERITIES for f in inline_findings
    )

    pr = state["pr_payload"]
    markdown = "\n".join(
        [
            f"## Automated Review — PR #{pr.pr_id}: {pr.title}",
            "",
            _section("Security", security),
            _section("Architecture", architecture),
            _section("ML/DS Anti-Patterns", ml_debugger),
            drift_text,
            f"**Overall: {'PASSED' if overall_passed else 'CHANGES REQUESTED'}**",
        ]
    )

    final_review = FinalPRReview(
        labels=_labels_for(security, architecture, ml_debugger, drift_flagged),
        overall_passed=overall_passed,
        markdown_summary=markdown,
        inline_findings=inline_findings,
    )

    return {"final_review": final_review}
