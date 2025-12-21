"""Utilities for managing the generation queue."""

from __future__ import annotations

import asyncio
from asyncio import QueueEmpty
from dataclasses import dataclass
from typing import Awaitable, Callable, List, Optional, Tuple, TYPE_CHECKING

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
        self._queue: "asyncio.PriorityQueue[tuple[int, int, QueuedGeneration]]" = asyncio.PriorityQueue()
        self._worker: Optional[asyncio.Task[None]] = None
        self._current: Optional[QueuedGeneration] = None
        self._counter: int = 0
        self._update_callback: Optional[Callable[[], Awaitable[None]]] = None

    def set_update_callback(self, callback: Callable[[], Awaitable[None]]) -> None:
        """Register a coroutine callback invoked whenever the queue changes."""

        self._update_callback = callback

    async def _notify_updated(self) -> None:
        if not self._update_callback:
            return

        try:
            await self._update_callback()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Queue update callback failed: %s", exc)

    async def add_to_queue(
        self,
        generation_func: Callable[..., Awaitable[None]],
        context: "GenerationContext",
        *args,
        priority: int = 0,
        **kwargs,
    ) -> None:
        """Add a new generation request to the queue."""

        job = QueuedGeneration(generation_func, args, kwargs, context)
        self._counter += 1
        await self._queue.put((-priority, self._counter, job))
        logger.info(
            "Queue: added request for user %s • size=%d • priority=%d",
            context.user_id,
            self._queue.qsize(),
            priority,
        )

        if not self._worker or self._worker.done():
            self._worker = asyncio.create_task(self._process_queue())

        await self._notify_updated()

    async def _process_queue(self) -> None:
        """Process queued generation requests sequentially."""

        while True:
            _, _, job = await self._queue.get()
            self._current = job
            await self._notify_updated()

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
                await self._notify_updated()

    async def cancel_pending(self, context: "GenerationContext") -> bool:
        """Remove a pending job from the queue if it hasn't started yet."""

        removed = False
        pending: list[tuple[int, int, QueuedGeneration]] = []

        while True:
            try:
                entry = self._queue.get_nowait()
            except QueueEmpty:
                break

            _, _, job = entry
            if job.context is context:
                removed = True
                logger.info(
                    "Queue: removed pending request for user %s",
                    context.user_id,
                )
                self._queue.task_done()
                continue

            pending.append(entry)
            self._queue.task_done()

        for entry in pending:
            self._queue.put_nowait(entry)

        if removed:
            await self._notify_updated()
        return removed

    def get_queue_position(self) -> int:
        """Get current queue size."""

        return self._queue.qsize()

    def get_pending_contexts(self) -> list["GenerationContext"]:
        """Return a snapshot of pending contexts in processing order."""

        entries = list(self._queue._queue)  # type: ignore[attr-defined]
        entries.sort()
        return [job.context for _, _, job in entries]

    def size(self, include_current: bool = True) -> int:
        """Return queue size, optionally counting the active item."""

        total = self._queue.qsize()
        if include_current and self._current:
            total += 1
        return total

    @property
    def current_context(self) -> Optional["GenerationContext"]:
        """Return the context that is currently being processed, if any."""

        return self._current.context if self._current else None
