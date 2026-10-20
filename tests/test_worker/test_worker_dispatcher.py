import asyncio

import pytest

from src.worker.dispatcher import DelayedTaskQueue


@pytest.mark.asyncio
async def test_schedule_runs_after_delay():
    ran = asyncio.Event()

    async def job():
        ran.set()

    q = DelayedTaskQueue()
    q.schedule(job, delay_seconds=0.01)

    assert not ran.is_set()  # hasn't fired yet
    await asyncio.sleep(0.05)
    assert ran.is_set()


@pytest.mark.asyncio
async def test_pending_count_reflects_in_flight_tasks():
    async def slow_job():
        await asyncio.sleep(0.05)

    q = DelayedTaskQueue()
    q.schedule(slow_job, delay_seconds=0)
    await asyncio.sleep(0)  # let the task start
    assert q.pending_count() == 1

    await asyncio.sleep(0.1)
    assert q.pending_count() == 0


@pytest.mark.asyncio
async def test_failing_job_does_not_raise_or_leak(caplog):
    async def bad_job():
        raise ValueError("boom")

    q = DelayedTaskQueue()
    q.schedule(bad_job, delay_seconds=0)
    await asyncio.sleep(0.05)

    assert q.pending_count() == 0  # failure was swallowed and logged, not left dangling


@pytest.mark.asyncio
async def test_schedule_every_repeats_and_survives_a_failing_run():
    calls = []

    async def job():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("first sweep fails")

    q = DelayedTaskQueue()
    q.schedule_every(job, interval_seconds=0.01)
    await asyncio.sleep(0.1)

    assert len(calls) >= 3
    assert q.pending_count() == 1  # still running

    q.cancel_all()
    await asyncio.sleep(0.01)  # cancellation lands, then its done-callback runs
    assert q.pending_count() == 0


# --- Celery backend (CELERY_BROKER_URL set) ---------------------------------

from dataclasses import replace  # noqa: E402

from src.worker import celery_app as celery_module  # noqa: E402
from src.worker import dispatcher, jobs  # noqa: E402


@pytest.fixture
def celery_on(monkeypatch):
    monkeypatch.setattr(dispatcher, "settings", replace(dispatcher.settings, celery_broker_url="redis://x/0"))


def test_schedule_sends_known_job_to_celery_with_countdown(celery_on, monkeypatch):
    sent = []
    monkeypatch.setattr(
        celery_module.pr_code_review, "apply_async", lambda **kw: sent.append(kw)
    )
    q = DelayedTaskQueue()
    q.schedule(
        jobs.run_pr_code_review,
        {"repo_full_name": "o/r", "pr_number": 3, "head_sha": "abc123"},
        delay_seconds=0,
    )
    assert sent == [{"kwargs": {"repo_full_name": "o/r", "pr_number": 3, "head_sha": "abc123"}, "countdown": 0}]
    assert q.pending_count() == 0  # nothing held in this process


def test_schedule_unknown_job_with_celery_fails_loudly(celery_on):
    async def mystery():
        pass

    with pytest.raises(ValueError, match="mystery"):
        DelayedTaskQueue().schedule(mystery, delay_seconds=0)


@pytest.mark.asyncio
async def test_schedule_every_defers_to_celery_beat(celery_on):
    q = DelayedTaskQueue()
    q.schedule_every(jobs.run_monitoring_sweep, interval_seconds=60)
    assert q.pending_count() == 0


def test_every_async_job_has_a_celery_task():
    # a new job added to jobs.py without a task would only fail once Celery is on
    async_jobs = {n for n in dir(jobs) if n.startswith("run_") and asyncio.iscoroutinefunction(getattr(jobs, n))}
    assert async_jobs == set(celery_module.TASKS)


def test_task_runs_the_job_body(monkeypatch):
    called = []
    monkeypatch.setattr(jobs, "_run_deployment_regression_check", lambda *a: called.append(a))
    celery_module.deployment_regression_check.apply(args=("o/r", "abc123", "prod")).get()
    assert called == [("o/r", "abc123", "prod")]


def test_monitoring_sweep_fans_out_one_task_per_model(monkeypatch):
    monkeypatch.setattr(jobs, "monitored_models", lambda: ("a", "b"))
    queued = []
    monkeypatch.setattr(celery_module.drift_check, "delay", queued.append)
    celery_module.monitoring_sweep.apply().get()
    assert queued == ["a", "b"]
