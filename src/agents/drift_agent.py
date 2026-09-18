# src/agents/drift_agent.py
"""Drift specialist: wraps drift/pr_drift.check_pr_drift as a LangGraph
node. This is the "statistical diagnostics on complex data" branch that
runs in parallel with the static-analysis specialists (security,
architecture, ml_debugger).

Needs a local git checkout to diff data files against (check_pr_drift
shells out to `git show`), so unlike the other specialists it's built via
a factory that closes over repo_path rather than being a bare function -
the same injected-dependency shape as agents/orchestrator.diagnose()
taking a retriever/store rather than constructing one itself."""

from __future__ import annotations

from pathlib import Path

from src.drift.pr_drift import check_pr_drift
from src.schemas.graph import AgentNode, AgentState


def make_drift_agent(repo_path: str | Path) -> AgentNode:
    def run_drift_agent(state: AgentState) -> dict:
        pr = state["pr_payload"]
        try:
            reports = check_pr_drift(pr, repo_path)
        except ValueError as exc:
            # No base_sha to compare against - not fatal to the rest of the
            # review, just means this branch has nothing to report.
            return {"drift_report": [], "errors": [f"drift_agent: {exc}"]}
        return {"drift_report": reports}

    return run_drift_agent
