# src/worker/dispatcher.py
"""Minimal asyncio-based delayed task queue. BluePrint.md is explicit:
'Simple asyncio task queue for v1; do not reach for Celery/Redis until you
have a reason (multi-instance deploy)'. Good enough for a single-process
devtool; not durable across process restarts - a scheduled task is lost if
the process dies before it fires."""

import asyncio
import logging
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class DelayedTaskQueue:
    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()

    def schedule(
        self,
        func: Callable[..., Awaitable[None]],
        kwargs: dict[str, Any] | None = None,
        *,
        delay_seconds: float,
    ) -> None:
        """Fire func(**kwargs) after delay_seconds, on the current event loop.

        Takes func + kwargs (not a closure) so the call shape matches a
        Celery task.apply_async(kwargs=..., countdown=...) - a lambda isn't
        picklable across a Celery/Redis broker, but a function reference and
        a plain kwargs dict are.
        """
        task = asyncio.create_task(self._run_after_delay(func, kwargs or {}, delay_seconds))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_after_delay(
        self, func: Callable[..., Awaitable[None]], kwargs: dict[str, Any], delay_seconds: float
    ) -> None:
        try:
            await asyncio.sleep(delay_seconds)
            await func(**kwargs)
        except Exception:
            logger.exception("Delayed task failed")

    def schedule_every(
        self,
        func: Callable[..., Awaitable[None]],
        kwargs: dict[str, Any] | None = None,
        *,
        interval_seconds: float,
    ) -> None:
        """Run func(**kwargs) every interval_seconds until the loop stops
        (Workflow 5's always-on sweep). A failing run is logged and the next
        one still fires - one bad sweep must not end monitoring."""
        task = asyncio.create_task(self._run_every(func, kwargs or {}, interval_seconds))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_every(
        self, func: Callable[..., Awaitable[None]], kwargs: dict[str, Any], interval_seconds: float
    ) -> None:
        while True:
            await self._run_after_delay(func, kwargs, interval_seconds)

    def cancel_all(self) -> None:
        """Stop every scheduled and recurring task (server shutdown). Pending
        delayed jobs are dropped - the same loss as a process restart."""
        for task in list(self._tasks):
            task.cancel()

    def pending_count(self) -> int:
        return len(self._tasks)


queue = DelayedTaskQueue()
