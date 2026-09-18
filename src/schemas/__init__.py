# src/schemas/__init__.py
#
# NOTE: this package currently holds schemas for two different product
# concepts that have not been reconciled - see Execution.md at the repo
# root for the full flag:
#   - pr.py, review.py, graph.py, webhook.py, linter.py,
#     rag.py: the PR-review-bot described in README.md (inline GitHub
#     review comments for ML/DS performance anti-patterns).
#   - telemetry.py: the "ML Reliability Copilot" (model-regression root
#     cause diagnosis) described in Architecture.md/BluePrint.md/PRD.md/
#     Workflow.md, dated 2026-09-04.
# pr.py/webhook.py/rag.py are plausibly reusable as the github/ evidence
# source for the new direction; review.py/graph.py/linter.py are specific
# to the old inline-PR-review product and are not used by the new one.
# Nothing has been deleted - confirm which product this repo is before
# removing either side.
from .pr import PRPayload, ChangedFile
from .review import ReviewFinding, AgentReviewReport, FinalPRReview, Severity, PRLabel
from .rag import CodeChunk, RetrievalResult
from .graph import AgentState
from .webhook import GitHubPullRequestEvent, WebhookPR, GitHubDeploymentStatusEvent
from .linter import LinterIssue, LinterReport
from .telemetry import NormalizedRun
from .drift import (
    ColumnDriftReport,
    ColumnKind,
    DatasetDriftReport,
    DriftMethod,
    MethodResult,
)
from .root_cause import DriftRootCause, FeatureAttribution, ShiftDirection

__all__ = [
    # PR & Diffs (PR-review-bot / potential github/ evidence source)
    "PRPayload",
    "ChangedFile",
    # Reviews & Labels (PR-review-bot only)
    "ReviewFinding",
    "AgentReviewReport",
    "FinalPRReview",
    "Severity",
    "PRLabel",
    # Retrieval
    "CodeChunk",
    "RetrievalResult",
    # LangGraph State (PR-review-bot only)
    "AgentState",
    # Webhooks & Tools
    "GitHubPullRequestEvent",
    "WebhookPR",
    "GitHubDeploymentStatusEvent",
    "LinterIssue",
    "LinterReport",
    # ML Reliability Copilot diagnostic core
    "NormalizedRun",
    # Data drift (Workflow 2)
    "DriftMethod",
    "ColumnKind",
    "MethodResult",
    "ColumnDriftReport",
    "DatasetDriftReport",
    # SHAP root-cause attribution
    "DriftRootCause",
    "FeatureAttribution",
    "ShiftDirection",
]
