# src/agents/multi_agent_graph.py
"""Builds the multi-agent PR review graph: security, architecture, and
ml_debugger run as static-analysis branches over the PR diff, drift runs
in parallel as the statistical-diagnostics branch over any changed data
files, and synthesize fans all four back into one FinalPRReview.

    START --> security ------\
          --> architecture ---+--> synthesize --> END
          --> ml_debugger ---/
          --> drift ---------/   (only when repo_path is given)

LangGraph schedules a node only once every edge into it has fired, so
synthesize always sees the finished output of every branch that ran -
this is what BluePrint.md's "agents/ orchestration" call means by running
specialists "simultaneously" rather than as a fixed pipeline.

security and ml_debugger are built via make_security_agent(repo_path) and
make_ml_debugger_agent(repo_path), same reason drift is built via
make_drift_agent(repo_path): their Groq tool-calling passes can read real
files (security's run_ast_security_scan; ml_debugger's
inspect_dataset_schema, run_drift_analysis, explain_drift_root_cause),
which only resolve correctly against a local PR checkout. Unlike drift,
their static regex/AST checks are useful with no checkout at all, so both
are always added as branches - repo_path is merely None there instead of
the whole node being skipped."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import cast

from langgraph.graph import END, START, StateGraph

from src.agents.architecture_agent import run_architecture_agent
from src.agents.drift_agent import make_drift_agent
from src.agents.ml_debugger_agent import make_ml_debugger_agent
from src.agents.security_agent import make_security_agent
from src.agents.synthesizer import run_synthesizer_agent
from src.schemas.graph import AgentNode, AgentState
from src.schemas.pr import PRPayload

logger = logging.getLogger(__name__)

# Maps each branch name to the AgentState field it writes to, so a branch
# that raises can still report an empty result there instead of taking
# down the whole graph run - see _isolate below. synthesize is isolated the
# same way, against "final_review" (empty=None, not []).
_OUTPUT_KEYS = {
    "architecture": "architecture_findings",
    "security": "security_findings",
    "ml_debugger": "ml_debugger_findings",
    "drift": "drift_report",
}


def _isolate(name: str, node: AgentNode, output_key: str, empty: object = None) -> AgentNode:
    """Wraps a node so an unexpected exception in it can't crash
    app.invoke() for the whole graph - it instead surfaces as `empty` for
    that node's output key plus an errors entry, which is what AgentState's
    docstring promises ("one slow/failing branch can't clobber another's
    results"). The fan-in is wrapped too: a raise in the synthesizer would
    otherwise throw away every branch's completed work."""

    def wrapped(state: AgentState) -> dict:
        try:
            return node(state)
        except Exception as exc:  # noqa: BLE001 - isolating an unknown failure on purpose
            logger.exception("%s node failed", name)
            return {output_key: empty, "errors": [f"{name}: {exc}"]}

    return wrapped


def build_pr_diff(pr: PRPayload) -> str:
    """Concatenates every changed file's patch into one unified-diff-style
    blob - the pr_diff seed value for the initial AgentState."""
    parts = []
    for f in pr.changed_files:
        parts.append(f"--- {f.filename} ({f.status})")
        if f.patch:
            parts.append(f.patch)
    return "\n".join(parts)


def build_multi_agent_graph(repo_path: str | Path | None = None):
    """Compiles the graph. repo_path is threaded into security and
    ml_debugger (so their tools can resolve real file paths) and into the
    drift branch (which needs a local git checkout to diff data files
    against at all - without repo_path, drift is skipped and drift_report
    stays whatever the initial state set it to)."""
    graph = StateGraph(AgentState)

    branches = {
        "architecture": run_architecture_agent,
        "security": make_security_agent(repo_path),
        "ml_debugger": make_ml_debugger_agent(repo_path),
    }
    if repo_path is not None:
        branches["drift"] = make_drift_agent(repo_path)

    for name, node in branches.items():
        graph.add_node(name, _isolate(name, node, _OUTPUT_KEYS[name], []))
    branch_names = list(branches)

    graph.add_node("synthesize", _isolate("synthesize", run_synthesizer_agent, "final_review"))

    for name in branch_names:
        graph.add_edge(START, name)
        graph.add_edge(name, "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


def initial_state(pr: PRPayload) -> AgentState:
    return AgentState(
        pr_payload=pr,
        pr_diff=build_pr_diff(pr),
        security_findings=[],
        architecture_findings=[],
        ml_debugger_findings=[],
        drift_report=[],
        final_review=None,
        errors=[],
    )


def run_pr_review(pr: PRPayload, repo_path: str | Path | None = None) -> AgentState:
    """Entry point: runs the full multi-agent graph for one PR and returns
    the final AgentState (state["final_review"] is the synthesized verdict).

    CompiledStateGraph.invoke()'s default (version="v1") overload is typed
    `-> dict[str, Any] | Any` regardless of the graph's state schema - only
    the version="v2" overload is generic over OutputT - so the cast here
    is langgraph's own documented typing boundary, not a workaround for a
    bug: at runtime this is exactly an AgentState dict, since that's the
    schema StateGraph(AgentState) was built with."""
    app = build_multi_agent_graph(repo_path)
    return cast(AgentState, app.invoke(initial_state(pr)))
