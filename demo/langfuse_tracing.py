# demo/langfuse_tracing.py
"""Wraps the multi-agent PR review graph with Langfuse tracing so a demo
can open the Langfuse UI and show the four specialist agents' execution
trace, concurrency, and timestamps - and, once a specialist is upgraded
from a heuristic to a real LLM call (see the note below), per-call token
usage and latency broken down by model/provider.

Deliberately kept out of src/ - the CLI (src/cli.py) and webhook path
(src/api/webhooks.py) do not need langfuse or the full langchain package
this needs (see demo/requirements-demo.txt); this is a presentation-layer
add-on for the "multi-agent observability" story, not part of the
diagnostic core.

Honest limitation: security_agent/architecture_agent/ml_debugger_agent/
drift_agent are all deterministic (regex/statistics), matching this
codebase's existing pattern of not calling an LLM without a configured
key (Execution.md). A trace from this module will therefore show four
real, concurrent node executions with accurate latency and timestamps,
but no token counts or Groq-vs-Gemini model spans - those only appear
once at least one node makes an actual LLM call traced by Langfuse
(e.g. via @observe or the langchain callback picking up a real
ChatGroq/ChatGoogleGenerativeAI invocation inside a node)."""

from __future__ import annotations

import os
from pathlib import Path

from langfuse import get_client
from langfuse.langchain import CallbackHandler

from src.agents.multi_agent_graph import build_multi_agent_graph, initial_state
from src.schemas.graph import AgentState
from src.schemas.pr import PRPayload


def langfuse_is_configured() -> bool:
    """True only if credentials are set AND the configured host accepts
    them - what to check before promising a demo audience a trace will
    actually land in the UI."""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return False
    return get_client().auth_check()


def run_traced_pr_review(pr: PRPayload, repo_path: str | Path | None = None) -> AgentState:
    """Runs the same graph as agents.multi_agent_graph.run_pr_review, with
    a Langfuse CallbackHandler attached so LangGraph's per-node
    start/end events (security, architecture, ml_debugger, drift running
    concurrently, then synthesize) are traced.

    Safe to call with no Langfuse credentials configured: the client logs
    an auth warning and disables itself rather than raising, so the graph
    still runs and returns a normal AgentState - you just won't see
    anything in the Langfuse UI until LANGFUSE_PUBLIC_KEY/
    LANGFUSE_SECRET_KEY (and LANGFUSE_HOST, if self-hosting) are set."""
    handler = CallbackHandler()
    app = build_multi_agent_graph(repo_path)
    result = app.invoke(
        initial_state(pr),
        config={
            "callbacks": [handler],
            "run_name": f"pr-review-{pr.repo_full_name}#{pr.pr_id}",
        },
    )
    # Langfuse batches events on a background thread; flush before the
    # process can exit so a short-lived demo script doesn't drop the trace.
    get_client().flush()
    return result
