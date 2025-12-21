from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

import discord


@dataclass(frozen=True)
class RoleTier:
    level: int
    name: str
    role_id: Optional[int]
    daily_limit: Optional[int]
    queue_priority: int
    max_parallel_generations: int


@dataclass
class GenerationContext:
    user_id: str
    user: discord.abc.User
    workflow_type: str
    is_donor: bool
    prompt: Optional[str] = None
    prompt_preset_name: Optional[str] = None
    prompt_preset_tags: Optional[str] = None
    model_preset_name: Optional[str] = None
    lora_preset_name: Optional[str] = None
    settings: Optional[str] = None
    resolution: Optional[str] = None
    seed: Optional[int] = None
    started_at: float = field(default_factory=time.time)
    workflow_name: Optional[str] = None
    message: Optional[discord.Message] = None
    prompt_id: Optional[str] = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    view: Optional[discord.ui.View] = None
    tier: RoleTier = field(default_factory=lambda: RoleTier(0, "Public", None, 25, 0, 1))
    daily_limit: Optional[int] = None
    counted_usage: bool = False
    slot_counted: bool = False
    completed: bool = False
    cancelled_notified: bool = False
    finalized: bool = False
    force_spoiler: bool = False
    processing: bool = False
