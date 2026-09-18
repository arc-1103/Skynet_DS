# graph.py
import operator
from typing import Annotated, List, Optional, Protocol, TypedDict
from src.schemas.drift import DatasetDriftReport
from src.schemas.pr import PRPayload
from src.schemas.review import FinalPRReview, ReviewFinding


class AgentState(TypedDict):
    """State for the multi-agent PR review graph (agents/multi_agent_graph.py).

    Fan-out/fan-in shape: security, architecture, ml_debugger and drift all
    run in parallel off the same pr_diff/pr_payload, each writing to its own
    named field (so one slow/failing branch can't clobber another's
    results), and synthesize merges all four into final_review.
    """

    # Base inputs
    pr_payload: PRPayload
    pr_diff: str

    # Parallel specialist outputs - operator.add merges concurrent branch
    # writes instead of one branch overwriting another's.
    security_findings: Annotated[List[ReviewFinding], operator.add]
    architecture_findings: Annotated[List[ReviewFinding], operator.add]
    ml_debugger_findings: Annotated[List[ReviewFinding], operator.add]
    # One entry per data file check_pr_drift could compare; empty means no
    # tabular data was touched (or drift wasn't checked), not an error.
    drift_report: List[DatasetDriftReport]

    # Synthesized output
    final_review: Optional[FinalPRReview]

    # Execution Metadata
    errors: Annotated[List[str], operator.add]


class AgentNode(Protocol):
    """Structural type for a specialist LangGraph node: `state ->
    partial-state-update dict`. Node factories (drift_agent.make_drift_agent,
    ml_debugger_agent.make_ml_debugger_agent) should annotate their return
    type with this, not `Callable[[AgentState], dict]` - langgraph's own
    StateNode type requires a callable whose parameter is named `state`,
    and `typing.Callable` always erases parameter names, so a
    `Callable[...]`-typed return fails that structural check in Pylance/
    pyright even though the actual function satisfies it fine at runtime.
    A Protocol keeps the name, matching langgraph's own node type exactly."""

    def __call__(self, state: AgentState) -> dict: ...
