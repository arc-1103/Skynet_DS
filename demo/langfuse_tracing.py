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

from src.agents.multi_agent_graph import run_pr_review
from src.schemas.graph import AgentState
from src.schemas.pr import PRPayload


def langfuse_is_configured() -> bool:
    """True only if credentials are set AND the configured host accepts
    them - what to check before promising a demo audience a trace will
    actually land in the UI."""
    if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
        return False
    try:
        return get_client().auth_check()
    except Exception:  # a rejected key raises (401) rather than returning False
        return False


def run_traced_pr_review(pr: PRPayload, repo_path: str | Path | None = None) -> AgentState:
    """Runs agents.multi_agent_graph.run_pr_review, which is itself traced
    (root span, one agent span per node, a generation per Groq call - see
    src/core/tracing.py), then flushes so a short-lived script keeps its trace.

    Safe to call with no Langfuse credentials configured: the client logs
    an auth warning and disables itself rather than raising, so the graph
    still runs and returns a normal AgentState - you just won't see
    anything in the Langfuse UI until LANGFUSE_PUBLIC_KEY/
    LANGFUSE_SECRET_KEY (and LANGFUSE_HOST, if self-hosting) are set."""
    result = run_pr_review(pr, repo_path)
    # Langfuse batches events on a background thread; flush before the
    # process can exit so a short-lived demo script doesn't drop the trace.
    get_client().flush()
    return result
