"""Utilities for managing the generation queue."""

from __future__ import annotations

import asyncio
from asyncio import QueueEmpty
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, Tuple, TYPE_CHECKING

from logger import logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.bot.imagesmith import GenerationContext


@dataclass
class QueuedGeneration:
    """A single queued generation job."""

    func: Callable[..., Awaitable[None]]
    args: Tuple
    kwargs: dict
    context: "GenerationContext"


class GenerationQueue:
    """Manages queued generation requests."""

    def __init__(self) -> None:
        self._queue: "asyncio.Queue[QueuedGeneration]" = asyncio.Queue()
        self._worker: Optional[asyncio.Task[None]] = None
        self._current: Optional[QueuedGeneration] = None

    async def add_to_queue(
        self,
        generation_func: Callable[..., Awaitable[None]],
        context: "GenerationContext",
        *args,
        **kwargs,
    ) -> None:
        """Add a new generation request to the queue."""

        job = QueuedGeneration(generation_func, args, kwargs, context)
        await self._queue.put(job)
        logger.info(
            "Queue: added request for user %s • size=%d",
            context.user_id,
            self._queue.qsize(),
        )

        if not self._worker or self._worker.done():
            self._worker = asyncio.create_task(self._process_queue())

    async def _process_queue(self) -> None:
        """Process queued generation requests sequentially."""

        while True:
            job = await self._queue.get()
            self._current = job

            if job.context.cancel_event.is_set():
                logger.info(
                    "Queue: skipping cancelled request for user %s",
                    job.context.user_id,
                )
                self._queue.task_done()
                continue

            logger.info(
                "Queue: processing request for user %s • remaining=%d",
                job.context.user_id,
                self._queue.qsize(),
            )

            try:
                await job.func(job.context, *job.args, **job.kwargs)
            except asyncio.CancelledError:  # pragma: no cover - defensive
                logger.info(
                    "Queue: worker cancelled while handling user %s",
                    job.context.user_id,
                )
                raise
            except Exception as exc:  # pragma: no cover - defensive
                logger.error(
                    "Queue: error while processing request for user %s: %s",
                    job.context.user_id,
                    exc,
                )
            finally:
                self._queue.task_done()
                self._current = None

    async def cancel_pending(self, context: "GenerationContext") -> bool:
        """Remove a pending job from the queue if it hasn't started yet."""

        removed = False
        pending: list[QueuedGeneration] = []

        while True:
            try:
                job = self._queue.get_nowait()
            except QueueEmpty:
                break

            if job.context is context:
                removed = True
                logger.info(
                    "Queue: removed pending request for user %s",
                    context.user_id,
                )
                self._queue.task_done()
                continue

            pending.append(job)
            self._queue.task_done()

        for job in pending:
            self._queue.put_nowait(job)

        return removed

    def get_queue_position(self) -> int:
        """Get current queue size."""

        return self._queue.qsize()

    @property
    def current_context(self) -> Optional["GenerationContext"]:
        """Return the context that is currently being processed, if any."""

        return self._current.context if self._current else None
