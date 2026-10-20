# src/worker/jobs.py
"""Job bodies scheduled by worker/dispatcher.py. Wires Workflow 2 (drift),
Workflow 4 (Deployment Regression Triage), Workflow 5 (the monitoring
sweep) and the PR-review graph into the diagnostic core built in agents/."""

import asyncio
import logging
import re
import subprocess
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.agents.multi_agent_graph import run_pr_review
from src.api.notify import notify_slack
from src.agents.orchestrator import diagnose, diagnose_drift, drift_signature
from src.core.config import settings
from src.drift.detector import detect_dataset_drift
from src.drift.pr_drift import data_files_in_pr
from src.drift.root_cause import explain_drift
from src.drift.tables import frame_from_features
from src.github_integration.pr_client import fetch_pr
from src.github_integration.repo_client import clone_or_checkout
from src.memory.sqlite_store import SQLiteStore
from src.observability.base import build_adapter
from src.observability.prometheus_adapter import PrometheusAdapter
from src.retrieval.evidence import EvidenceRetriever
from src.schemas.webhook import SHA_PATTERN

logger = logging.getLogger(__name__)

# Anything used to build a filename has to be a plain hex ref.
_SHA = re.compile(SHA_PATTERN)


async def run_deployment_regression_check(repo_full_name: str, deploy_sha: str, environment: str) -> None:
    """Workflow 4 steps 2-5: fetch the model's latest run, compare to its
    baseline, rank causes scoped to the deploy's commit range.

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
    model_id = settings.repo_models.get(repo_full_name)
    if model_id is None:
        logger.warning(
            "deploy of %s has no model mapped in MLCOP_REPO_MODELS - skipping regression check",
            repo_full_name,
        )
        return
    logger.info(
        "deployment regression check: repo=%s sha=%s env=%s model=%s",
        repo_full_name, deploy_sha, environment, model_id,
    )
    with closing(SQLiteStore(settings.memory_db_path)) as store:
        retriever = EvidenceRetriever(
            build_adapter(settings.telemetry_backend, settings.wandb_entity),
            store,
            # the deploy's own repo, not GITHUB_REPO: the commit range is this repo's
            github_repo=repo_full_name,
            github_token=settings.github_token,
        )
        recent = retriever.get_recent_runs(model_id, limit=1)
        if not recent:
            logger.warning("no runs found for %s - nothing to diagnose", model_id)
            return
        result = diagnose(recent[0].run_id, retriever, store, deploy_sha=deploy_sha)

    logger.info("deployment regression check result for %s: %s", repo_full_name, result.explanation)
    if not result.ranked_candidates:  # no metric moved - not an incident, don't page
        return
    notify_slack(
        settings.slack_webhook_url,
        f"Deploy {deploy_sha[:7]} of {repo_full_name} to {environment} - {model_id} post-deploy check",
        result.explanation,
    )


async def run_drift_check(model_id: str) -> None:
    """Off the event loop, for the same reason as the deploy check above -
    the SHAP surrogate fit in diagnose_drift is CPU work."""
    await asyncio.to_thread(check_drift, model_id)


def check_drift(model_id: str, now: datetime | None = None) -> str | None:
    """Workflow 2 steps 1-3: pull the live feature window from Prometheus,
    compare it to the baseline distribution in memory/, and hand a drift to
    agents/ for diagnosis. The first check for a model has nothing to
    compare against, so it records the window as the baseline instead.

    Diagnosis and the Slack page happen once per drift *episode*: a sweep
    that sees the same features still drifted is the same incident, not a
    new one, and a clean check closes the episode. Re-diagnosing every sweep
    paged every 15 minutes and, worse, counted each sweep as a recurrence -
    one persistent drift read as "seasonal" within four sweeps.

    Returns the explanation when a new episode was diagnosed, else None."""
    if not settings.prometheus_url:
        logger.warning("PROMETHEUS_URL not set - drift check for %s skipped", model_id)
        return None
    now = now or datetime.now(timezone.utc)
    current = PrometheusAdapter(settings.prometheus_url).get_feature_samples(
        model_id, now - timedelta(seconds=settings.drift_window_seconds), now
    )
    if not current.features:
        logger.warning(
            "no feature samples for %s in the window - is it exporting mlcop_feature_value?", model_id
        )
        return None

    with closing(SQLiteStore(settings.memory_db_path)) as store:
        baseline = store.get_baseline_distribution(model_id)
        if baseline is None:
            store.set_baseline_distribution(model_id, current)
            logger.info("captured the drift baseline for %s (%s features)", model_id, len(current.features))
            return None

        report = detect_dataset_drift(
            frame_from_features(baseline.features), frame_from_features(current.features), label=model_id
        )
        _publish(model_id, "drift", report)
        active = store.get_active_drift(model_id)
        if not report.dataset_drifted:
            _publish(model_id, "root_cause", None)  # a stale attribution would outlive its drift
            if active:
                store.set_active_drift(model_id, None)
                logger.info("drift episode for %s resolved (%s)", model_id, active)
            logger.info("no drift for %s: %s/%s features drifted", model_id, report.n_drifted, report.n_columns)
            return None

        signature = drift_signature(report)
        if signature == active:
            logger.info("drift for %s is still ongoing (%s) - already reported", model_id, signature)
            return None

        try:
            # ponytail: a second SHAP fit per episode (diagnose_drift runs its own);
            # return the attribution from diagnose_drift if the fit cost matters
            _publish(model_id, "root_cause", explain_drift(
                frame_from_features(baseline.features), frame_from_features(current.features), label=model_id
            ))
        except ValueError:  # too few rows to fit the surrogate, same case diagnose_drift tolerates
            pass

        # the model's own repo if a deploy mapping names one, else GITHUB_REPO
        repo = next((r for r, m in settings.repo_models.items() if m == model_id), settings.github_repo)
        retriever = EvidenceRetriever(
            build_adapter(settings.telemetry_backend, settings.wandb_entity), store, repo, settings.github_token
        )
        result = diagnose_drift(baseline, current, report, retriever, store, settings.pipeline_paths)
        store.set_active_drift(model_id, signature)

    # Workflow 2 step 6: flag + explanation out to the log and Slack
    logger.warning("drift diagnosed for %s:\n%s", model_id, result.explanation)
    notify_slack(settings.slack_webhook_url, f"Drift detected on {model_id}", result.explanation)
    return result.explanation


def _publish(model_id: str, kind: str, payload) -> None:
    """Leave the model's latest `kind` ("drift" | "root_cause") where the
    dashboard reads it; None removes it. Best effort - the dashboard is a
    view, so a full or read-only disk must not fail the drift check."""
    path = Path(settings.monitoring_dir) / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', model_id)}.{kind}.json"
    try:
        if payload is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    except OSError:
        logger.warning("could not publish %s for %s to %s", kind, model_id, path, exc_info=True)


def monitored_models() -> tuple[str, ...]:
    return settings.monitored_models or tuple(dict.fromkeys(settings.repo_models.values()))


async def run_monitoring_sweep() -> None:
    """Workflow 5: one pass over every watched model. Pure scheduling - the
    diagnosis is Workflow 2's. Deploy regressions are already queued per
    deploy by the deployment_status webhook (Workflow 4), so the sweep does
    not re-check them. One model's failure doesn't stop the rest."""
    for model_id in monitored_models():
        try:
            await run_drift_check(model_id)
        except Exception:
            logger.exception("drift check failed for %s", model_id)


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
