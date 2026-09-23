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
