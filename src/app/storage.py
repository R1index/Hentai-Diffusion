from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from logger import logger


@dataclass
class GenerationStats:
    total: int
    daily: Dict[str, int]


class GenerationStore:
    """Persistence layer for generation counts and stats."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.counts: Dict[str, int] = defaultdict(int)
        self.stats: Dict[str, GenerationStats] = {}
        self.last_reset: float = time.time()
        self._load()

    # ---------------------- CRUD ----------------------
    def increment(self, user_id: str) -> None:
        self.counts[user_id] = self.counts.get(user_id, 0) + 1
        self._save()

    def decrement(self, user_id: str) -> None:
        self.counts[user_id] = max(0, self.counts.get(user_id, 0) - 1)
        self._save()

    def set_count(self, user_id: str, value: int) -> None:
        self.counts[user_id] = max(0, int(value))
        self._save()

    def get_count(self, user_id: str) -> int:
        return int(self.counts.get(user_id, 0))

    def record_success(self, user_id: str, *, counted_usage: bool) -> None:
        """Record a successful generation. If counted_usage=False, only stats are updated."""

        delta = 1
        if counted_usage:
            # Counts were incremented earlier; keep stats in sync but do not modify counts.
            delta = 1
        self._touch_stat(user_id, delta=delta)
        self._save()

    def reset_daily_if_needed(self) -> bool:
        if time.time() - self.last_reset >= 86400:
            self.counts = defaultdict(int)
            self.last_reset = time.time()
            self._save()
            logger.info("Daily generation counters reset")
            return True
        return False

    def summary(self, user_id: str, *, retention_days: int = 90) -> Dict[str, int]:
        self._prune_history(retention_days)
        stats = self.stats.get(user_id)
        if not stats:
            return {"day": 0, "week": 0, "month": 0, "total": 0}

        today = time.strftime("%Y-%m-%d", time.gmtime())
        day_total = 0
        week_total = 0
        month_total = 0
        total = int(stats.total)
        for date_str, count in stats.daily.items():
            try:
                ts = time.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                continue
            diff_days = (time.mktime(time.gmtime()) - time.mktime(ts)) / 86400
            if diff_days < 1:
                day_total += count
            if diff_days < 7:
                week_total += count
            if diff_days < 30:
                month_total += count

        return {"day": day_total, "week": week_total, "month": month_total, "total": total}

    # ---------------------- Internal ----------------------
    def _touch_stat(self, user_id: str, *, delta: int) -> None:
        today_key = time.strftime("%Y-%m-%d", time.gmtime())
        stats = self.stats.setdefault(user_id, GenerationStats(total=0, daily={}))
        stats.total = int(stats.total) + delta
        stats.daily[today_key] = int(stats.daily.get(today_key, 0)) + delta
        self._prune_history()

    def _prune_history(self, retention_days: int = 90) -> None:
        threshold_ts = time.time() - retention_days * 86400
        for user_id, stats in list(self.stats.items()):
            for key in list(stats.daily.keys()):
                try:
                    ts = time.mktime(time.strptime(key, "%Y-%m-%d"))
                except ValueError:
                    stats.daily.pop(key, None)
                    continue
                if ts < threshold_ts or ts > time.time():
                    stats.daily.pop(key, None)
            if not stats.daily and stats.total == 0:
                self.stats.pop(user_id, None)

    def _load(self) -> None:
        try:
            if not self.path.exists():
                self._save()
                return
            with self.path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self.counts = defaultdict(int, data.get("counts", {}))
            self.last_reset = data.get("last_reset", time.time())
            raw_stats = data.get("stats") or {}
            self._load_stats(raw_stats)
            logger.info("Generation counters restored from disk")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load generation counts: %s", exc)
            self.counts = defaultdict(int)
            self.stats = {}
            self.last_reset = time.time()
            self._save()

    def _load_stats(self, raw_stats: dict) -> None:
        self.stats = {}
        for user_id, payload in raw_stats.items():
            if not isinstance(payload, dict):
                continue
            total = int(payload.get("total", 0) or 0)
            daily_raw = payload.get("daily") or {}
            if not isinstance(daily_raw, dict):
                daily_raw = {}
            daily: Dict[str, int] = {}
            for date_str, count in daily_raw.items():
                try:
                    daily[str(date_str)] = int(count)
                except (TypeError, ValueError):
                    continue
            self.stats[str(user_id)] = GenerationStats(total=total, daily=daily)
        self._prune_history()

    def _serialize_stats(self) -> Dict[str, Dict[str, Any]]:
        serialized: Dict[str, Dict[str, Any]] = {}
        for user_id, stats in self.stats.items():
            daily = {k: int(v) for k, v in (stats.daily or {}).items()}
            serialized[user_id] = {"total": int(stats.total), "daily": daily}
        return serialized

    def _save(self) -> None:
        data = {
            "counts": dict(self.counts),
            "last_reset": self.last_reset,
            "stats": self._serialize_stats(),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, allow_unicode=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to save generation counts: %s", exc)
