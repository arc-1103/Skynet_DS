# src/worker/celery_app.py
"""Durable worker: the same job bodies as worker/jobs.py, run by Celery so a
queued job survives an API restart and the post-deploy check's 1-hour wait
isn't held in one process's memory (the asyncio queue loses both).

Opt-in via CELERY_BROKER_URL; dispatcher.queue hands jobs here when it is
set. Run a worker (-B embeds the scheduler that fires the monitoring sweep):

    celery -A src.worker.celery_app worker -B -l info

Tasks take the jobs' plain kwargs so they serialize as JSON."""

from celery import Celery

from src.core.config import settings
from src.worker import jobs

# "memory://" only so importing this module without a broker (tests) works;
# nothing sends to it unless CELERY_BROKER_URL is set.
celery_app = Celery("skynet", broker=settings.celery_broker_url or "memory://")
celery_app.conf.update(
    task_serializer="json",
    # Long CPU-bound jobs: take one at a time, and re-queue it if the worker
    # dies mid-run rather than losing it.
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    # Redis re-delivers a task not acked within visibility_timeout (default
    # 1h) - exactly the deploy check's countdown, so it would run twice.
    broker_transport_options={"visibility_timeout": 4 * 60 * 60},
)
if settings.monitor_interval_seconds > 0 and settings.prometheus_url:
    celery_app.conf.beat_schedule = {
        "monitoring-sweep": {"task": "skynet.monitoring_sweep", "schedule": settings.monitor_interval_seconds}
    }


@celery_app.task(name="skynet.pr_code_review")
def pr_code_review(repo_full_name: str, pr_number: int, head_sha: str) -> None:
    jobs._run_pr_code_review(repo_full_name, pr_number, head_sha)


@celery_app.task(name="skynet.deployment_regression_check")
def deployment_regression_check(repo_full_name: str, deploy_sha: str, environment: str) -> None:
    jobs._run_deployment_regression_check(repo_full_name, deploy_sha, environment)


@celery_app.task(name="skynet.drift_check")
def drift_check(model_id: str) -> None:
    jobs.check_drift(model_id)


@celery_app.task(name="skynet.monitoring_sweep")
def monitoring_sweep() -> None:
    """Fan out one drift_check per model so one model's failure (or a slow
    SHAP fit) neither blocks nor takes down the rest."""
    for model_id in jobs.monitored_models():
        drift_check.delay(model_id)


# Keyed by the asyncio job's function name, which is how dispatcher.queue
# knows a job it was handed has a Celery equivalent.
TASKS = {
    jobs.run_pr_code_review.__name__: pr_code_review,
    jobs.run_deployment_regression_check.__name__: deployment_regression_check,
    jobs.run_drift_check.__name__: drift_check,
    jobs.run_monitoring_sweep.__name__: monitoring_sweep,
}
