# src/worker/dispatcher.py
"""Minimal asyncio-based delayed task queue. BluePrint.md is explicit:
'Simple asyncio task queue for v1; do not reach for Celery/Redis until you
have a reason (multi-instance deploy)'. Good enough for a single-process
devtool; not durable across process restarts - a scheduled task is lost if
the process dies before it fires."""

import asyncio
import logging
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)


class DelayedTaskQueue:
    def __init__(self) -> None:
        self._tasks: set[asyncio.Task] = set()

    def schedule(self, coro_factory: Callable[[], Awaitable[None]], delay_seconds: float) -> None:
        """Fire coro_factory() after delay_seconds, on the current event loop."""
        task = asyncio.create_task(self._run_after_delay(coro_factory, delay_seconds))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _run_after_delay(self, coro_factory: Callable[[], Awaitable[None]], delay_seconds: float) -> None:
        try:
            await asyncio.sleep(delay_seconds)
            await coro_factory()
        except Exception:
            logger.exception("Delayed task failed")

    def pending_count(self) -> int:
        return len(self._tasks)


queue = DelayedTaskQueue()
