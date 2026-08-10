# graph.py
import operator
from typing import Annotated, Dict, List, Optional, TypedDict
from src.schemas.pr import PRPayload
from src.schemas.review import AgentReviewReport, FinalPRReview, PRLabel, ReviewFinding


class PRReviewState(TypedDict):
    """
    State object passed across LangGraph nodes.
    Accumulated fields use operator.add to prevent parallel branch overwrites.
    """
    # Base Inputs
    pr_payload: PRPayload
    raw_diff: str

    # Extracted AST Context
    modified_symbols: List[str]                  # Function/Class names extracted via Tree-sitter
    retrieved_code_context: Dict[str, str]        # File path -> Retrieved Code Chunks

    # Parallel Sub-Agent Outputs (Merged using operator.add)
    findings: Annotated[List[ReviewFinding], operator.add]
    agent_reports: Annotated[List[AgentReviewReport], operator.add]

    # Synthesized Output
    suggested_labels: List[PRLabel]
    final_review: Optional[FinalPRReview]

    # Execution Metadata
    errors: Annotated[List[str], operator.add]