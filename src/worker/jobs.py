# src/worker/jobs.py
"""Job bodies scheduled by worker/dispatcher.py. Wires Workflow 4
(Deployment Regression Triage, Workflow.md) into the diagnostic core built
in agents/orchestrator.py."""

import asyncio
import logging
import re
import subprocess
from pathlib import Path

from src.agents.multi_agent_graph import run_pr_review
from src.agents.orchestrator import diagnose
from src.core.config import settings
from src.drift.pr_drift import data_files_in_pr
from src.github_integration.pr_client import fetch_pr
from src.github_integration.repo_client import clone_or_checkout
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

# Anything used to build a filename has to be a plain hex ref - a webhook
# payload can claim any `sha` it likes, including "../../../../x".
_SHA = re.compile(r"[0-9a-fA-F]{4,64}")


async def run_deployment_regression_check(repo_full_name: str, deploy_sha: str, environment: str) -> None:
    """Workflow 4 steps 2-5: fetch post-deploy metrics, compare to baseline,
    rank causes scoped to the deploy. Currently runs against fake data - see
    the TODO above.

    Every job body here is synchronous (GitHub REST calls, `git clone`,
    pandas/sklearn) and these coroutines are scheduled onto the API
    server's own event loop by worker/dispatcher.py, so the work runs in a
    thread: left inline it blocks the loop for as long as the clone and
    the drift fit take, and every concurrent webhook delivery then blows
    past GitHub's 10s timeout."""
    await asyncio.to_thread(
        _run_deployment_regression_check, repo_full_name, deploy_sha, environment
    )


def _run_deployment_regression_check(repo_full_name: str, deploy_sha: str, environment: str) -> None:
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


async def run_pr_code_review(repo_full_name: str, pr_number: int, head_sha: str) -> None:
    """Runs _run_pr_code_review off the event loop - see
    run_deployment_regression_check's docstring for why."""
    await asyncio.to_thread(_run_pr_code_review, repo_full_name, pr_number, head_sha)


def _run_pr_code_review(repo_full_name: str, pr_number: int, head_sha: str) -> None:
    """Webhook -> worker step for the 'pull_request' event: pull the PR's
    metadata and diff from GitHub, run the multi-agent review over it, and
    cache both under settings.pr_cache_dir.

    A local clone is only made when the PR touches a tabular data file: the
    drift branch is the one specialist that can't work off the diff alone
    (it shells out to `git show` for the base version of each file), and
    it's also what makes repo_path worth paying a clone for - with it, the
    security and ml_debugger tool passes can read real files too. Without
    a data file, which is the common case, the graph runs repo_path=None
    and drift is skipped rather than the whole review being skipped.

    Nothing is posted back to GitHub yet - the review is written to disk
    for the (not yet built) PR-comment step, which is an open decision
    about app identity and permissions."""
    if not _SHA.fullmatch(head_sha):
        raise ValueError(f"refusing to build a cache path from non-SHA head_sha {head_sha!r}")

    logger.info(
        "fetching PR for local review: repo=%s pr=%s sha=%s", repo_full_name, pr_number, head_sha
    )
    pr = fetch_pr(repo_full_name, pr_number, settings.github_token)

    out_dir = Path(settings.pr_cache_dir) / repo_full_name.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pr_{pr_number}_{head_sha}.json"
    out_path.write_text(pr.model_dump_json(indent=2), encoding="utf-8")

    logger.info("cached PR #%s (%s) locally at %s", pr_number, repo_full_name, out_path)

    repo_path = None
    data_files = data_files_in_pr(pr)
    if data_files:
        logger.info("PR #%s touches %s data file(s); cloning for drift checks", pr_number, len(data_files))
        try:
            repo_path = clone_or_checkout(
                repo_full_name,
                pr.commit_sha,
                settings.github_token,
                extra_refs=(pr.base_sha,) if pr.base_sha else (),
            )
        except (subprocess.CalledProcessError, OSError, TimeoutError) as exc:
            # A transient git failure, a missing git executable, or a busy repo
            # lock shouldn't cost the whole review: the PR is already cached and
            # the static-analysis branches work off the diff alone. Degrade to a
            # review without a checkout instead of letting dispatcher.py's
            # blanket handler log this as an opaque job failure.
            logger.warning(
                "clone failed for PR #%s, reviewing without a checkout (drift skipped): %s",
                pr_number,
                exc,
            )

    state = run_pr_review(pr, repo_path)

    for error in state.get("errors", []):
        logger.warning("PR #%s review branch error: %s", pr_number, error)

    for report in state.get("drift_report", []):
        # The label is built from PR file paths, which git allows to contain a
        # literal backslash, a colon, and anything else that isn't "/" - and
        # with_name() raises on an OS separator. Everything outside a safe set
        # collapses to "_"; the dot is kept so existing cache names are stable.
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(report.label))
        out_path.with_name(f"pr_{pr_number}_{head_sha}_drift_{slug}.json").write_text(
            report.model_dump_json(indent=2), encoding="utf-8"
        )
        logger.info(
            "drift in %s: %s/%s columns drifted (%.0f%%) - dataset_drifted=%s",
            report.label,
            report.n_drifted,
            report.n_columns,
            report.share_drifted * 100,
            report.dataset_drifted,
        )

    review = state.get("final_review")
    if review is None:
        # _isolate caught a raise in the synthesizer; the branches' own
        # output is already logged/persisted above.
        logger.error("PR #%s produced no final review", pr_number)
        return

    review_path = out_path.with_name(f"pr_{pr_number}_{head_sha}_review.json")
    review_path.write_text(review.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "reviewed PR #%s: %s, %s inline finding(s), labels=%s -> %s",
        pr_number,
        "PASSED" if review.overall_passed else "CHANGES REQUESTED",
        len(review.inline_findings),
        [label.value for label in review.labels],
        review_path,
    )
