import time
from pathlib import Path

import pytest

from src.app.storage import GenerationStore


@pytest.fixture()
def store(tmp_path: Path) -> GenerationStore:
    return GenerationStore(tmp_path / "counts.yml")


def test_increment_and_reset(store: GenerationStore):
    store.increment("u1")
    assert store.get_count("u1") == 1
    store.reset_daily_if_needed()
    # Force reset by manipulating timestamp
    store.last_reset -= 90000
    assert store.reset_daily_if_needed() is True
    assert store.get_count("u1") == 0


def test_record_success_stats(store: GenerationStore):
    # Unlimited user: only stats
    store.record_success("u2", counted_usage=False)
    summary = store.summary("u2")
    assert summary["total"] == 1
    # Limited user: count + stats
    store.increment("u3")
    store.record_success("u3", counted_usage=True)
    summary3 = store.summary("u3")
    assert store.get_count("u3") == 1
    assert summary3["total"] == 1
