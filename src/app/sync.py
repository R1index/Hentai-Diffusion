from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from typing import Any, Dict, Optional

import discord

from logger import logger
from .models import RoleTier
from .config import SyncConfig


class SyncBridge:
    """Encapsulates cross-bot synchronization over a Discord channel."""

    def __init__(self, bot: discord.Client, config: SyncConfig) -> None:
        self.bot = bot
        self.config = config
        self._seen: Dict[str, float] = {}
        self._remote_slots: Dict[str, int] = {}
        self._remote_expiry: Dict[str, float] = {}

    # ---------------------- Signing ----------------------
    def _canon(self, obj: dict) -> str:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    def _sign(self, payload: dict) -> str:
        msg = self._canon(payload).encode("utf-8")
        key = self.config.shared_secret.encode("utf-8")
        return hmac.new(key, msg, hashlib.sha256).hexdigest()

    def _verify(self, wrapper: dict) -> bool:
        try:
            expected = self._sign(wrapper["payload"])
            sig = wrapper["sig"]
            return hmac.compare_digest(expected, sig)
        except Exception:
            return False

    def _now(self) -> float:
        return time.time()

    def prune_seen(self) -> None:
        now = self._now()
        for key, expiry in list(self._seen.items()):
            if expiry <= now:
                self._seen.pop(key, None)

    # ---------------------- Remote slots ----------------------
    def prune_remote_slots(self) -> None:
        now = self._now()
        for uid, expiry in list(self._remote_expiry.items()):
            if expiry <= now:
                self._remote_expiry.pop(uid, None)
                self._remote_slots.pop(uid, None)

    def remote_active(self, user_id: str) -> int:
        self.prune_remote_slots()
        return max(0, int(self._remote_slots.get(user_id, 0)))

    # ---------------------- Publishing ----------------------
    async def publish_limit(
        self,
        *,
        channel: discord.abc.Messageable,
        user_id: int,
        used: int,
        limit: Optional[int],
        reset_at: float,
        tier: RoleTier,
    ) -> None:
        payload = {
            "kind": "limit",
            "event_id": str(uuid.uuid4()),
            "ts": int(self._now()),
            "source_bot_id": self.bot.user.id if self.bot.user else 0,
            "user_id": int(user_id),
            "generations_used": int(used),
            "limit": None if limit is None else int(limit),
            "reset_at": float(reset_at),
            "tier": {
                "name": tier.name,
                "limit": None if limit is None else int(limit),
                "queue_priority": int(tier.queue_priority),
                "max_parallel": int(tier.max_parallel_generations),
            },
        }
        wrapper = {"payload": payload, "sig": self._sign(payload)}
        await channel.send(self.config.prefix + self._canon(wrapper))
        self._seen[payload["event_id"]] = self._now() + self.config.ttl_seconds
        self.prune_seen()

    async def publish_active_delta(
        self,
        *,
        channel: discord.abc.Messageable,
        user_id: int,
        delta: int,
        tier: RoleTier,
    ) -> None:
        payload = {
            "kind": "active_delta",
            "event_id": str(uuid.uuid4()),
            "ts": int(self._now()),
            "source_bot_id": self.bot.user.id if self.bot.user else 0,
            "user_id": int(user_id),
            "delta": int(delta),
            "tier": {
                "name": tier.name,
                "limit": None,
                "queue_priority": int(tier.queue_priority),
                "max_parallel": int(tier.max_parallel_generations),
            },
        }
        wrapper = {"payload": payload, "sig": self._sign(payload)}
        await channel.send(self.config.prefix + self._canon(wrapper))
        self._seen[payload["event_id"]] = self._now() + self.config.ttl_seconds
        self.prune_seen()

    # ---------------------- Consumption ----------------------
    def handle_sync_payload(self, wrapper: dict, *, source_bot_id: int) -> Optional[dict]:
        if not self._verify(wrapper):
            logger.warning("SYNC signature mismatch — ignored")
            return None

        payload = wrapper["payload"]
        event_id = payload["event_id"]
        kind = payload.get("kind", "limit")
        if self.bot.user and source_bot_id == self.bot.user.id:
            return None

        self.prune_seen()
        if event_id in self._seen:
            return None
        self._seen[event_id] = self._now() + self.config.ttl_seconds

        if kind == "active_delta":
            user_id = str(int(payload["user_id"]))
            delta = int(payload.get("delta", 0))
            current = self.remote_active(user_id)
            updated = max(0, current + delta)
            if updated:
                self._remote_slots[user_id] = updated
                self._remote_expiry[user_id] = self._now() + self.config.ttl_seconds
            else:
                self._remote_slots.pop(user_id, None)
                self._remote_expiry.pop(user_id, None)
            return {"kind": "active_delta", "user_id": user_id, "delta": delta, "total": updated}

        return {"kind": "limit", "payload": payload}
