import asyncio
import pytest

from src.core.generation_queue import GenerationQueue, QueuedGeneration


@pytest.mark.asyncio
async def test_queue_priority_order():
    queue = GenerationQueue()
    seen = []

    class DummyCtx:
        def __init__(self, uid: str):
            self.user_id = uid
            self.cancel_event = asyncio.Event()

    async def worker(ctx, label):
        seen.append(label)

    async def enqueue():
        await queue.add_to_queue(worker, DummyCtx("1"), priority=0, *["a"])
        await queue.add_to_queue(worker, DummyCtx("1"), priority=10, *["b"])
        await queue.add_to_queue(worker, DummyCtx("1"), priority=5, *["c"])
        # Give worker time
        await asyncio.sleep(0.2)
        # Cancel worker task to exit loop if still running
        if queue._worker and not queue._worker.done():
            queue._worker.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queue._worker

    await enqueue()
    assert seen == ["b", "c", "a"]
