# src/worker/jobs.py
"""Job bodies scheduled by worker/dispatcher.py. Wires Workflow 4
(Deployment Regression Triage, Workflow.md) into the diagnostic core built
in agents/orchestrator.py."""

import logging

from src.agents.orchestrator import diagnose
from src.core.config import settings
from src.memory.sqlite_store import SQLiteStore
from src.observability.fake_adapter import FakeTelemetryAdapter
from src.retrieval.evidence import EvidenceRetriever

logger = logging.getLogger(__name__)

# TODO(Phase 1, BluePrint.md section 5): there is no real mapping yet from a
# GitHub repo/deployment environment to a model_id/run_id in observability/,
# and no real MLflow/W&B adapter - both are open decisions the user hasn't
# made. Until then this job resolves against the fake adapter's one seeded
# model so the webhook -> async dispatch -> diagnose wiring can be exercised
# end-to-end; it does not check anything real yet.
_STUB_MODEL_ID = "fraud-clf"


async def run_deployment_regression_check(repo_full_name: str, deploy_sha: str, environment: str) -> None:
    """Workflow 4 steps 2-5: fetch post-deploy metrics, compare to baseline,
    rank causes scoped to the deploy. Currently runs against fake data - see
    the TODO above."""
    logger.info(
        "deployment regression check: repo=%s sha=%s env=%s (stubbed against fake adapter)",
        repo_full_name,
        deploy_sha,
        environment,
    )
    adapter = FakeTelemetryAdapter()
    store = SQLiteStore(settings.memory_db_path)
    retriever = EvidenceRetriever(adapter, store)

    recent = retriever.get_recent_runs(_STUB_MODEL_ID, limit=1)
    if not recent:
        logger.warning("no runs found for %s - nothing to diagnose", _STUB_MODEL_ID)
        return

    result = diagnose(recent[0].run_id, retriever, store)
    logger.info("deployment regression check result for %s: %s", repo_full_name, result.explanation)
