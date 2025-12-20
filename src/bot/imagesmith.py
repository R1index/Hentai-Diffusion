from __future__ import annotations
import asyncio
import hashlib
import hmac
import importlib
import inspect
import json
import os
import re
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import discord
import yaml
from discord import app_commands
from discord.ext import commands, tasks

from logger import logger
from .commands import profile_command, rgen_command, workflows_command
from ..comfy.client import ComfyUIClient
from ..comfy.workflow_manager import WorkflowManager
from ..core.generation_queue import GenerationQueue
from ..core.hook_manager import HookManager
from ..core.plugin import Plugin
from ..core.security import BasicSecurity, SecurityManager
from ..ui import embeds as ui_embeds
from ..ui.views import GenerationView


@dataclass
@dataclass(frozen=True)
class RoleTier:
    """Represents an access tier driven by Discord roles."""

    level: int
    name: str
    role_id: Optional[int]
    daily_limit: Optional[int]
    queue_priority: int
    max_parallel_generations: int


@dataclass
class GenerationContext:
    """State for a single in-flight generation."""

    user_id: str
    user: discord.abc.User
    workflow_type: str
    is_donor: bool
    prompt: Optional[str] = None
    prompt_preset_name: Optional[str] = None
    settings: Optional[str] = None
    resolution: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    workflow_name: Optional[str] = None
    message: Optional[discord.Message] = None
    prompt_id: Optional[str] = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    view: Optional[GenerationView] = None
    tier: RoleTier = field(default_factory=lambda: RoleTier(0, "Public", None, 25, 0, 1))
    daily_limit: Optional[int] = None
    counted_usage: bool = False
    slot_counted: bool = False
    completed: bool = False
    cancelled_notified: bool = False
    finalized: bool = False
    force_spoiler: bool = False


class ComfyUIBot(commands.Bot):
    GENERATION_COUNTS_FILE = "generation_counts.yml"
    QUEUE_STUCK_THRESHOLD = 1800  # 30 minutes
    STATS_RETENTION_DAYS = 90

    def __init__(self, configuration_path: str = "configuration.yml", plugins_path: str = "plugins"):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.guilds = True

        super().__init__(command_prefix="/", intents=intents)

        self.configuration_path = configuration_path
        self.plugins_path = plugins_path

        # Core components
        self.workflow_manager = WorkflowManager(configuration_path)
        self.security_manager = SecurityManager()
        self.hook_manager = HookManager()
        self.basic_security = BasicSecurity(self)
        self.comfy_client: Optional[ComfyUIClient] = None
        self.generation_queue = GenerationQueue()

        # Plugin system
        self.plugins: List[Plugin] = []

        # Generation tracking
        self.active_generations: Dict[str, List[GenerationContext]] = defaultdict(list)
        self.synced_active_slots: Dict[str, int] = defaultdict(int)
        self._sync_channel_cache: Optional[discord.abc.Messageable] = None

        # Spoiler handling
        self._spoiler_tags: Set[str] = set()
        self._spoiler_tag_display: Dict[str, str] = {}

        # --- Sync over Discord channel (both bots publish & listen) ---
        self.SYNC_CHANNEL_ID: int = 1406937891203977358  # TODO: set your hidden #bot-sync channel ID
        self.SYNC_SHARED_SECRET: str = "fj39f9aj2J#d9a!_38dja0d9@qwe93"  # same secret on both bots
        self.SYNC_PREFIX: str = "SYNC "
        self._sync_seen: Dict[str, float] = {}
        self._sync_seen_ttl: float = 3600.0  # 1 hour

        # Donation system
        self.blocked_users: Set[str] = set()
        self.donor_users: Set[str] = set()
        self.access_guild_id: Optional[str] = None
        self.supporter_role_name: str = "Supporter"
        self.supporter_role_id: Optional[str] = None
        self.role_tiers: List[RoleTier] = [
            RoleTier(
                level=4,
                name="Level 4",
                role_id=1451769149045997588,
                daily_limit=None,
                queue_priority=40,
                max_parallel_generations=30,
            ),
            RoleTier(
                level=3,
                name="Level 3",
                role_id=1451768900453925030,
                daily_limit=None,
                queue_priority=30,
                max_parallel_generations=3,
            ),
            RoleTier(
                level=2,
                name="Level 2",
                role_id=1361296590777745560,
                daily_limit=None,
                queue_priority=0,
                max_parallel_generations=1,
            ),
            RoleTier(
                level=1,
                name="Level 1",
                role_id=1450781064418299914,
                daily_limit=100,
                queue_priority=0,
                max_parallel_generations=1,
            ),
            RoleTier(
                level=0,
                name="Public",
                role_id=None,
                daily_limit=25,
                queue_priority=0,
                max_parallel_generations=1,
            ),
        ]
        self.donor_tier = RoleTier(
            level=2,
            name="Donor",
            role_id=None,
            daily_limit=None,
            queue_priority=0,
            max_parallel_generations=1,
        )
        self.user_generation_counts: Dict[str, int] = defaultdict(int)
        self.user_generation_stats: Dict[str, Dict[str, Any]] = {}
        self.last_reset_time: float = time.time()

        os.makedirs("data", exist_ok=True)
        self._load_security_lists()
        self._load_generation_counts()

        self.monitor_generations.start()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _get_counts_path(self) -> str:
        return os.path.join("data", self.GENERATION_COUNTS_FILE)

    def _read_configuration_file(self) -> dict:
        """Load the YAML configuration, handling possible encodings."""

        for encoding in ("utf-8", "utf-8-sig", "cp1251"):
            try:
                with open(self.configuration_path, "r", encoding=encoding) as file:
                    return yaml.safe_load(file) or {}
            except FileNotFoundError:
                logger.warning("Configuration file %s not found; using defaults", self.configuration_path)
                return {}
            except UnicodeDecodeError:
                continue

        raise UnicodeDecodeError("auto", b"", 0, 1, "Unable to decode config in known encodings")

    def _write_configuration_file(self, data: dict) -> None:
        """Persist configuration data back to disk."""

        with open(self.configuration_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)

    def _normalize_tag(self, tag: str) -> str:
        return re.sub(r"\s+", " ", tag.strip()).lower()

    def _set_spoiler_tags(self, tags: Iterable[str]) -> None:
        self._spoiler_tags.clear()
        self._spoiler_tag_display.clear()

        for tag in tags:
            tag_str = str(tag).strip()
            if not tag_str:
                continue
            normalized = self._normalize_tag(tag_str)
            self._spoiler_tags.add(normalized)
            self._spoiler_tag_display[normalized] = tag_str

        if self._spoiler_tags:
            logger.info("Spoiler tags loaded: %s", ", ".join(sorted(self._spoiler_tag_display.values())))
        else:
            logger.info("No spoiler tags configured")

    def _persist_spoiler_tags(self) -> None:
        data = self._read_configuration_file()
        spoilers = data.setdefault("spoilers", {})
        spoilers["tags"] = sorted(self._spoiler_tag_display.values(), key=str.lower)
        self._write_configuration_file(data)

    def _load_security_lists(self) -> None:
        try:
            config_data = self._read_configuration_file()

            security = config_data.get("security", {}) or {}
            self.blocked_users = {str(user) for user in security.get("blocked_users", [])}
            self.donor_users = {str(user) for user in security.get("donor_users", [])}

            access_guild_id = security.get("access_guild_id")
            self.access_guild_id = str(access_guild_id) if access_guild_id else None
            self.supporter_role_name = security.get("supporter_role_name", "Supporter")

            supporter_role_id = security.get("supporter_role_id")
            self.supporter_role_id = str(supporter_role_id) if supporter_role_id else None

            self._set_spoiler_tags(config_data.get("spoilers", {}).get("tags", []))

            logger.info(
                "Security configuration loaded • blocked=%d donors=%d access_guild=%s",
                len(self.blocked_users),
                len(self.donor_users),
                self.access_guild_id or "disabled",
            )

        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load security configuration: %s", exc)
            self.blocked_users = set()
            self.donor_users = set()
            self.access_guild_id = None
            self.supporter_role_name = "Supporter"
            self.supporter_role_id = None
            self._set_spoiler_tags([])

    def _load_generation_counts(self) -> None:
        counts_path = self._get_counts_path()
        try:
            if os.path.exists(counts_path):
                with open(counts_path, "r", encoding="utf-8") as file:
                    data = yaml.safe_load(file) or {}
                    self.user_generation_counts = defaultdict(int, data.get("counts", {}))
                    self.last_reset_time = data.get("last_reset", time.time())
                    stats_mutated = self._load_generation_stats(data.get("stats"))
                    if stats_mutated:
                        self._save_generation_counts()
                    logger.info("Generation counters restored from disk")
            else:
                logger.info("Generation counters not found, creating new file")
                self._save_generation_counts()
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load generation counts: %s", exc)
            self.user_generation_counts = defaultdict(int)
            self.last_reset_time = time.time()
            self._save_generation_counts()

    def _save_generation_counts(self) -> None:
        counts_path = self._get_counts_path()
        try:
            data = {
                "counts": dict(self.user_generation_counts),
                "last_reset": self.last_reset_time,
                "stats": self._serialize_generation_stats(),
            }
            with open(counts_path, "w", encoding="utf-8") as file:
                yaml.safe_dump(data, file, allow_unicode=True)
            logger.debug("Generation counters saved")
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to save generation counts: %s", exc)

    def _reset_counts_if_needed(self) -> None:
        current_time = time.time()
        if current_time - self.last_reset_time >= 86400:
            self.user_generation_counts.clear()
            self.last_reset_time = current_time
            self._save_generation_counts()
            logger.info("Daily generation counters reset")

    def _get_spoiler_tag_list(self) -> List[str]:
        return sorted(self._spoiler_tag_display.values(), key=str.lower)

    def _prompt_contains_spoiler_tag(self, prompt: Optional[str]) -> bool:
        if not prompt or not self._spoiler_tags:
            return False

        normalized_prompt = re.sub(r"\s+", " ", prompt.lower())
        return any(tag in normalized_prompt for tag in self._spoiler_tags)

    def _format_user_for_log(self, user: discord.abc.User) -> str:
        base_name = (
            getattr(user, "global_name", None)
            or getattr(user, "display_name", None)
            or getattr(user, "name", None)
            or str(user)
        )
        return f"{base_name} ({getattr(user, 'id', '?')})"

    def _get_active_contexts(self, user_id: str) -> List[GenerationContext]:
        contexts = self.active_generations.get(user_id, [])
        return [ctx for ctx in contexts if not ctx.finalized]

    def _get_remote_active_slots(self, user_id: str) -> int:
        return max(0, int(self.synced_active_slots.get(user_id, 0)))

    async def _ensure_manage_guild(self, interaction: discord.Interaction) -> bool:
        perms = getattr(interaction.user, "guild_permissions", None)
        if perms and perms.manage_guild:
            return True

        embed = ui_embeds.build_limit_embed(
            "You need the **Manage Server** permission to modify spoiler tags."
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)
        return False

    def _load_generation_stats(self, raw_stats: Optional[dict]) -> bool:
        self.user_generation_stats = {}
        if not raw_stats:
            return False

        mutated = False
        for user_id, payload in raw_stats.items():
            if not isinstance(payload, dict):
                continue

            total = payload.get("total", 0)
            try:
                total_int = int(total)
            except (TypeError, ValueError):
                total_int = 0

            daily_raw = payload.get("daily") or {}
            if not isinstance(daily_raw, dict):
                daily_raw = {}
                mutated = True
            daily: Dict[str, int] = {}
            for date_str, count in daily_raw.items():
                try:
                    daily[str(date_str)] = int(count)
                except (TypeError, ValueError):
                    mutated = True

            if self._prune_daily_history(daily):
                mutated = True

            self.user_generation_stats[str(user_id)] = {
                "total": total_int,
                "daily": daily,
            }

        return mutated

    def _serialize_generation_stats(self) -> Dict[str, Dict[str, Any]]:
        serialized: Dict[str, Dict[str, Any]] = {}
        for user_id, payload in self.user_generation_stats.items():
            total = payload.get("total", 0)
            try:
                total_int = int(total)
            except (TypeError, ValueError):
                total_int = 0

            daily_src = payload.get("daily") or {}
            if not isinstance(daily_src, dict):
                daily_src = {}
            daily: Dict[str, int] = {}
            for date_str, count in daily_src.items():
                try:
                    daily[str(date_str)] = int(count)
                except (TypeError, ValueError):
                    continue

            serialized[user_id] = {
                "total": total_int,
                "daily": daily,
            }

        return serialized

    def _current_utc_date(self) -> date:
        return datetime.now(timezone.utc).date()

    def _prune_daily_history(self, history: Dict[str, int]) -> bool:
        if not history:
            return False

        today = self._current_utc_date()
        threshold = today - timedelta(days=self.STATS_RETENTION_DAYS)
        modified = False

        for key in list(history.keys()):
            try:
                day = datetime.strptime(key, "%Y-%m-%d").date()
            except ValueError:
                history.pop(key, None)
                modified = True
                continue

            if day < threshold or day > today:
                history.pop(key, None)
                modified = True

        return modified

    def _record_successful_generation(self, user_id: str) -> None:
        stats = self.user_generation_stats.setdefault(user_id, {"total": 0, "daily": {}})
        stats["total"] = int(stats.get("total", 0)) + 1
        daily = stats.setdefault("daily", {})
        if not isinstance(daily, dict):
            daily = {}
            stats["daily"] = daily
        today_key = self._current_utc_date().isoformat()
        daily[today_key] = int(daily.get(today_key, 0)) + 1
        self._prune_daily_history(daily)
        self._save_generation_counts()

    def get_user_generation_summary(self, user_id: str) -> Dict[str, int]:
        stats = self.user_generation_stats.get(user_id)
        if not stats:
            return {"day": 0, "week": 0, "month": 0, "total": 0}

        daily = stats.get("daily", {})
        changed = False
        if not isinstance(daily, dict):
            daily = {}
            stats["daily"] = daily
            changed = True

        if self._prune_daily_history(daily):
            changed = True

        if changed:
            self._save_generation_counts()

        today = self._current_utc_date()
        week_cutoff = today - timedelta(days=6)
        month_cutoff = today - timedelta(days=29)

        day_total = 0
        week_total = 0
        month_total = 0

        for key, value in daily.items():
            try:
                day = datetime.strptime(key, "%Y-%m-%d").date()
            except ValueError:
                continue

            count = int(value)
            if day == today:
                day_total += count
            if week_cutoff <= day <= today:
                week_total += count
            if month_cutoff <= day <= today:
                month_total += count

        total = int(stats.get("total", 0))
        return {
            "day": day_total,
            "week": week_total,
            "month": month_total,
            "total": total,
        }

    def _format_time_remaining(self) -> str:
        remaining = max(0.0, 86400 - (time.time() - self.last_reset_time))
        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)
        return f"{hours}h {minutes}m"

    def _get_public_tier(self) -> RoleTier:
        for tier in self.role_tiers:
            if tier.level == 0:
                return tier
        return RoleTier(0, "Public", None, 25, 0, 1)

    async def _get_access_guild(self) -> Optional[discord.Guild]:
        if not self.access_guild_id:
            logger.debug("access-guild: skip (id empty)")
            return None

        try:
            gid = int(self.access_guild_id)
        except Exception:
            logger.warning("access-guild: invalid id=%r", self.access_guild_id)
            return None

        guild = self.get_guild(gid)
        if guild:
            return guild

        try:
            guild = await self.fetch_guild(gid)
            logger.debug("access-guild: fetched guild %s", gid)
            return guild
        except Exception as exc:
            logger.warning("access-guild: fetch_guild(%s) failed: %s", gid, exc)
            return None

    async def _get_access_member(self, interaction: discord.Interaction) -> Optional[discord.Member]:
        try:
            guild = await self._get_access_guild()
            if not guild:
                return None

            if interaction.guild and interaction.guild.id == guild.id:
                target_guild = interaction.guild
                logger.debug("access-member: using interaction guild %s", guild.id)
            else:
                target_guild = guild

            member = target_guild.get_member(interaction.user.id)
            if member:
                return member

            try:
                member = await target_guild.fetch_member(interaction.user.id)
                logger.debug("access-member: fetched member %s on guild %s", interaction.user.id, target_guild.id)
                return member
            except Exception as exc:
                logger.info("access-member: user %s not in guild %s (%s)", interaction.user.id, target_guild.id, exc)
                return None
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("access-member: failed %s", exc)
            return None

    async def _determine_user_tier(self, interaction: discord.Interaction) -> RoleTier:
        highest = self._get_public_tier()
        member = await self._get_access_member(interaction)
        role_ids: Set[int] = {r.id for r in getattr(member, "roles", [])} if member else set()

        for tier in self.role_tiers:
            if tier.role_id and tier.role_id in role_ids and tier.level > highest.level:
                highest = tier

        if str(interaction.user.id) in self.donor_users and highest.level < self.donor_tier.level:
            highest = self.donor_tier

        return highest

    # ------------------------------------------------------------------
    # Background monitoring
    # ------------------------------------------------------------------
    @tasks.loop(minutes=5)
    async def monitor_generations(self) -> None:
        """Cancel generations that appear to be stuck."""

        current_time = time.time()
        stuck_contexts = [
            context
            for contexts in list(self.active_generations.values())
            for context in contexts
            if current_time - context.started_at > self.QUEUE_STUCK_THRESHOLD
        ]

        for context in stuck_contexts:
            logger.warning(
                "gen[%s|%s] timed out — cancelling",
                context.user_id,
                self._format_user_for_log(context.user),
            )
            context.cancel_event.set()
            if self.comfy_client and context.prompt_id:
                try:
                    await self.comfy_client.cancel_prompt(context.prompt_id)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("Timeout cancellation failed: %s", exc)
            await self._handle_cancelled_generation(context, reason="Generation timed out.")
            self._finalize_generation_context(context, success=False)

    @monitor_generations.before_loop
    async def before_monitor(self) -> None:
        await self.wait_until_ready()

    async def close(self) -> None:  # type: ignore[override]
        self.monitor_generations.cancel()
        self._save_generation_counts()
        await super().close()

    # ------------------------------------------------------------------
    # Discord lifecycle
    # ------------------------------------------------------------------
    async def setup_hook(self) -> None:
        logger.info("Setting up bot…")
        await self._load_plugins()

        try:
            await self.hook_manager.execute_hook(
                "is.comfyui.client.before_create",
                self.workflow_manager.config["comfyui"]["instances"],
            )

            self.comfy_client = ComfyUIClient(
                self.workflow_manager.config["comfyui"]["instances"],
                self.hook_manager,
            )

            await self.hook_manager.execute_hook(
                "is.comfyui.client.after_create",
                self.workflow_manager.config["comfyui"]["instances"],
            )

            await self.comfy_client.connect()
            logger.info("Connected to ComfyUI")
        except Exception as exc:
            logger.error("Failed to connect to ComfyUI: %s", exc)
            await self._cleanup_resources()
            sys.exit(1)

        logger.info("Registering slash commands")
        try:
            self.tree.add_command(rgen_command(self))
            self.tree.add_command(workflows_command(self))
            self.tree.add_command(self._create_limits_command())
            self.tree.add_command(self._create_spoiler_command())
            self.tree.add_command(profile_command(self))

            synced_commands = await self.tree.sync()
            command_list = ", ".join(f"/{cmd.name}" for cmd in synced_commands) or "none"
            logger.info("Slash commands synced: %s", command_list)
        except Exception as exc:
            logger.error("Failed to sync commands: %s", exc)
            await self._cleanup_resources()
            sys.exit(1)

    async def on_message(self, message: discord.Message) -> None:
        if not message.author.bot:
            return
        if message.channel.id != self.SYNC_CHANNEL_ID:
            return
        if not message.content.startswith(self.SYNC_PREFIX):
            return

        try:
            data = json.loads(message.content[len(self.SYNC_PREFIX) :])
            if not self._sync_verify(data):
                logger.warning("SYNC signature mismatch — ignored")
                return

            payload = data["payload"]
            event_id = payload["event_id"]
            src_bot = int(payload["source_bot_id"])
            kind = payload.get("kind", "limit")

            self._sync_prune_seen()
            if event_id in self._sync_seen:
                return
            self._sync_seen[event_id] = self._sync_now() + self._sync_seen_ttl

            if self.user and src_bot == self.user.id:
                return

            user_id = str(int(payload["user_id"]))
            tier_info = payload.get("tier", {})
            tier_name = tier_info.get("name", "unknown")
            queue_priority = tier_info.get("queue_priority", 0)
            max_parallel = tier_info.get("max_parallel", 1)

            if kind == "active_delta":
                delta = int(payload.get("delta", 0))
                current = self._get_remote_active_slots(user_id)
                updated = max(0, current + delta)
                if updated:
                    self.synced_active_slots[user_id] = updated
                else:
                    self.synced_active_slots.pop(user_id, None)

                logger.info(
                    "SYNC active slots update from bot %s → user %s tier=%s delta=%s total_remote=%s (max_parallel=%s)",
                    src_bot,
                    user_id,
                    tier_name,
                    delta,
                    updated,
                    max_parallel,
                )
                return

            used = int(payload["generations_used"])
            reset_at = float(payload["reset_at"])
            limit_raw = tier_info.get("limit", payload.get("limit"))
            limit = None if limit_raw in (None, -1) else int(limit_raw)
            incoming_last_reset = reset_at - 86400.0

            if getattr(self, "last_reset_time", 0) > incoming_last_reset + 2:
                logger.debug("SYNC skipped — local reset is newer")
                return

            if limit is None:
                logger.debug(
                    "SYNC ignored for unlimited tier • bot=%s user=%s tier=%s priority=%s max_parallel=%s",
                    src_bot,
                    user_id,
                    tier_name,
                    queue_priority,
                    max_parallel,
                )
                return

            self.user_generation_counts[user_id] = used
            self.last_reset_time = incoming_last_reset
            self._save_generation_counts()

            logger.info(
                "SYNC applied from bot %s → user %s tier=%s: %s/%s (priority=%s max_parallel=%s)",
                src_bot,
                user_id,
                tier_name,
                used,
                limit,
                queue_priority,
                max_parallel,
            )

        except Exception as exc:  # pragma: no cover - defensive
            logger.error("SYNC parse/apply failed: %s", exc, exc_info=True)

    async def on_ready(self) -> None:
        logger.info("Logged in as %s (ID: %s)", self.user, self.user.id if self.user else "?")
        logger.info("Connected to %d guilds", len(self.guilds))

        permissions = discord.Permissions(
            send_messages=True,
            read_messages=True,
            attach_files=True,
            embed_links=True,
            use_external_emojis=True,
            add_reactions=True,
            read_message_history=True,
        )

        invite_link = discord.utils.oauth_url(
            self.user.id,
            permissions=permissions,
            scopes=("bot", "applications.commands"),
        )

        logger.info("Invite link: %s", invite_link)
        logger.info("Bot is ready")

    async def _cleanup_resources(self) -> None:
        logger.info("Cleaning up resources…")
        try:
            if self.comfy_client:
                await self.comfy_client.close()
            await self.close()
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Error during cleanup: %s", exc)

    async def _load_plugins(self) -> None:
        plugins_dir = Path(self.plugins_path)
        if not plugins_dir.exists():
            logger.info("Plugins directory not found — skipping")
            return

        sys.path.append(str(Path.cwd()))
        plugin_files = [f for f in plugins_dir.glob("*.py") if f.name != "__init__.py"]

        for plugin_file in plugin_files:
            logger.info("Loading plugin: %s", plugin_file)
            try:
                spec = importlib.util.spec_from_file_location(plugin_file.stem, plugin_file)
                if spec is None or spec.loader is None:
                    logger.warning("Failed to load plugin spec: %s", plugin_file)
                    continue

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                logger.debug("Module loaded: %s", module.__name__)

                for item_name in dir(module):
                    if item_name.startswith("__"):
                        continue

                    try:
                        obj = getattr(module, item_name)
                        if inspect.isclass(obj) and issubclass(obj, Plugin) and obj is not Plugin:
                            plugin_instance = obj(self)
                            await plugin_instance.on_load()
                            self.plugins.append(plugin_instance)
                            logger.info("Plugin ready: %s", obj.__name__)
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.error("Error loading plugin item %s: %s", item_name, exc)
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to load plugin %s: %s", plugin_file, exc)

        logger.info("Plugins loaded: %d", len(self.plugins))

    # ------------------------------------------------------------------
    # Sync helpers
    # ------------------------------------------------------------------
    def _sync_now(self) -> float:
        return time.time()

    def _sync_canon(self, obj: dict) -> str:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    def _sync_sign(self, payload: dict) -> str:
        msg = self._sync_canon(payload).encode("utf-8")
        key = self.SYNC_SHARED_SECRET.encode("utf-8")
        return hmac.new(key, msg, hashlib.sha256).hexdigest()

    def _sync_verify(self, wrapper: dict) -> bool:
        try:
            payload = wrapper["payload"]
            sig = wrapper["sig"]
            expected = self._sync_sign(payload)
            return hmac.compare_digest(sig, expected)
        except Exception:
            return False

    def _sync_prune_seen(self) -> None:
        now = self._sync_now()
        for key, expiry in list(self._sync_seen.items()):
            if expiry <= now:
                self._sync_seen.pop(key, None)

    async def _get_sync_channel(self) -> Optional[discord.abc.Messageable]:
        if self._sync_channel_cache:
            return self._sync_channel_cache

        channel = self.get_channel(self.SYNC_CHANNEL_ID)
        if not channel:
            try:
                channel = await self.fetch_channel(self.SYNC_CHANNEL_ID)
            except Exception:
                channel = None

        if channel:
            self._sync_channel_cache = channel

        return channel

    async def _publish_limit_update(
        self,
        *,
        user_id: int,
        used: int,
        limit: Optional[int],
        reset_at: float,
        tier_name: str,
        queue_priority: int,
        max_parallel: int,
        ) -> None:
        try:
            channel = self.get_channel(self.SYNC_CHANNEL_ID)
            if not channel:
                try:
                    channel = await self.fetch_channel(self.SYNC_CHANNEL_ID)
                except Exception:
                    channel = None
            if not channel:
                return

            event_id = str(uuid.uuid4())
            payload = {
                "kind": "limit",
                "event_id": event_id,
                "ts": int(self._sync_now()),
                "source_bot_id": self.user.id if self.user else 0,
                "user_id": int(user_id),
                "generations_used": int(used),
                "limit": None if limit is None else int(limit),
                "reset_at": float(reset_at),
                "tier": {
                    "name": tier_name,
                    "limit": None if limit is None else int(limit),
                    "queue_priority": int(queue_priority),
                    "max_parallel": int(max_parallel),
                },
            }
            wrapper = {"payload": payload, "sig": self._sync_sign(payload)}
            await channel.send(self.SYNC_PREFIX + self._sync_canon(wrapper))
            self._sync_seen[event_id] = self._sync_now() + self._sync_seen_ttl
            self._sync_prune_seen()
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("publish_limit_update failed: %s", exc, exc_info=True)

    async def _publish_active_delta(
        self,
        *,
        user_id: int,
        delta: int,
        tier_name: str,
        queue_priority: int,
        max_parallel: int,
    ) -> None:
        try:
            channel = await self._get_sync_channel()
            if not channel or not self.SYNC_CHANNEL_ID:
                return

            event_id = str(uuid.uuid4())
            payload = {
                "kind": "active_delta",
                "event_id": event_id,
                "ts": int(self._sync_now()),
                "source_bot_id": self.user.id if self.user else 0,
                "user_id": int(user_id),
                "delta": int(delta),
                "tier": {
                    "name": tier_name,
                    "limit": None,
                    "queue_priority": int(queue_priority),
                    "max_parallel": int(max_parallel),
                },
            }
            wrapper = {"payload": payload, "sig": self._sync_sign(payload)}
            await channel.send(self.SYNC_PREFIX + self._sync_canon(wrapper))
            self._sync_seen[event_id] = self._sync_now() + self._sync_seen_ttl
            self._sync_prune_seen()
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("publish_active_delta failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------
    def _create_limits_command(self):
        @app_commands.command(name="limits", description="Check your current generation limits and status")
        async def limits_command(interaction: discord.Interaction) -> None:
            user_id = str(interaction.user.id)
            self._reset_counts_if_needed()

            tier = await self._determine_user_tier(interaction)
            is_donor = tier.daily_limit is None
            count = self.user_generation_counts.get(user_id, 0)
            queue_info = f"Queue slots: **{tier.max_parallel_generations}** at once."
            priority_hint = "Priority over lower tiers." if tier.queue_priority >= 30 else "Standard queue priority."

            if is_donor:
                description = (
                    f"🌟 **{tier.name}**: unlimited access!\n"
                    f"{queue_info}\n"
                    f"{priority_hint}\n"
                    "Thank you for supporting the project."
                )
                embed = ui_embeds.build_notice_embed(
                    title="💎 Unlimited access",
                    description=description,
                    color=ui_embeds.SUCCESS_COLOR,
                )
            else:
                usage = ui_embeds.format_usage_bar(
                    count,
                    tier.daily_limit or 0,
                    reset_hint=f"resets in {self._format_time_remaining()}",
                )
                description = (
                    f"🔒 You are using the {tier.name} tier.\n"
                    f"{queue_info}\n"
                    f"{priority_hint}\n"
                    "Support us to unlock unlimited generations!"
                )
                embed = ui_embeds.build_notice_embed(
                    title="📊 Daily usage",
                    description=f"{description}\n\n{usage}",
                    color=ui_embeds.WARNING_COLOR,
                )

            await interaction.response.send_message(embed=embed, ephemeral=True)

        return limits_command

    def _create_spoiler_command(self) -> app_commands.Group:
        spoiler_group = app_commands.Group(name="spoiler", description="Manage spoiler tags")

        @spoiler_group.command(name="list", description="Show configured spoiler tags")
        async def spoiler_list(interaction: discord.Interaction) -> None:
            tags = self._get_spoiler_tag_list()
            if tags:
                description = "\n".join(f"• `{tag}`" for tag in tags)
                title = "📑 Spoiler tags"
                color = ui_embeds.ACCENT_COLOR
            else:
                description = "No spoiler tags are configured. Use /spoiler add to create one."
                title = "ℹ️ No spoiler tags"
                color = ui_embeds.WARNING_COLOR

            embed = ui_embeds.build_notice_embed(
                title=title,
                description=description,
                color=color,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)

        @spoiler_group.command(name="add", description="Add a tag that will force image spoilers")
        @app_commands.describe(tag="Tag to treat as a spoiler trigger")
        async def spoiler_add(interaction: discord.Interaction, tag: str) -> None:
            if not await self._ensure_manage_guild(interaction):
                return

            cleaned = tag.strip()
            if not cleaned:
                embed = ui_embeds.build_notice_embed(
                    title="❌ Invalid tag",
                    description="Provide a non-empty tag to add.",
                    color=ui_embeds.ERROR_COLOR,
                )
                await interaction.response.send_message(embed=embed, ephemeral=True)
                return

            normalized = self._normalize_tag(cleaned)
            if normalized in self._spoiler_tags:
                existing = self._spoiler_tag_display.get(normalized, cleaned)
                embed = ui_embeds.build_notice_embed(
                    title="ℹ️ Tag already exists",
                    description=f"`{existing}` is already configured as a spoiler tag.",
                    color=ui_embeds.WARNING_COLOR,
                )
                await interaction.response.send_message(embed=embed, ephemeral=True)
                return

            self._spoiler_tags.add(normalized)
            self._spoiler_tag_display[normalized] = cleaned

            try:
                self._persist_spoiler_tags()
            except Exception as exc:  # pragma: no cover - defensive
                self._spoiler_tags.discard(normalized)
                self._spoiler_tag_display.pop(normalized, None)
                logger.error("Failed to persist spoiler tags: %s", exc)
                embed = ui_embeds.build_notice_embed(
                    title="❌ Failed to save tag",
                    description="The tag could not be saved. Check logs for details.",
                    color=ui_embeds.ERROR_COLOR,
                )
                await interaction.response.send_message(embed=embed, ephemeral=True)
                return

            embed = ui_embeds.build_notice_embed(
                title="✅ Spoiler tag added",
                description=f"Images will now be hidden behind spoilers when prompts include `{cleaned}`.",
                color=ui_embeds.SUCCESS_COLOR,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)

        @spoiler_group.command(name="remove", description="Remove a configured spoiler tag")
        @app_commands.describe(tag="Tag to remove from the spoiler list")
        async def spoiler_remove(interaction: discord.Interaction, tag: str) -> None:
            if not await self._ensure_manage_guild(interaction):
                return

            normalized = self._normalize_tag(tag)
            if normalized not in self._spoiler_tags:
                embed = ui_embeds.build_notice_embed(
                    title="❌ Tag not found",
                    description=f"`{tag}` is not configured as a spoiler tag.",
                    color=ui_embeds.ERROR_COLOR,
                )
                await interaction.response.send_message(embed=embed, ephemeral=True)
                return

            removed_display = self._spoiler_tag_display.get(normalized, tag)
            self._spoiler_tags.discard(normalized)
            self._spoiler_tag_display.pop(normalized, None)

            try:
                self._persist_spoiler_tags()
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to persist spoiler tags: %s", exc)
                # Revert in-memory state
                self._spoiler_tags.add(normalized)
                self._spoiler_tag_display[normalized] = removed_display
                embed = ui_embeds.build_notice_embed(
                    title="❌ Failed to remove tag",
                    description="The tag could not be removed. Check logs for details.",
                    color=ui_embeds.ERROR_COLOR,
                )
                await interaction.response.send_message(embed=embed, ephemeral=True)
                return

            embed = ui_embeds.build_notice_embed(
                title="🗑️ Spoiler tag removed",
                description=f"`{removed_display}` will no longer force spoilered images.",
                color=ui_embeds.SUCCESS_COLOR,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)

        return spoiler_group
    async def handle_generation(
        self,
        interaction: discord.Interaction,
        workflow_type: str,
        prompt: str,
        workflow: Optional[str] = None,
        settings: Optional[str] = None,
        resolution: Optional[str] = None,
        prompt_preset: Optional[str] = None,
        input_image: Optional[discord.Attachment] = None,
    ) -> None:
        user_id = str(interaction.user.id)
        self._reset_counts_if_needed()

        user_global_name = getattr(interaction.user, "global_name", None)
        logger.info("gen[%s] global_name=%s", user_id, user_global_name or "—")

        if user_id in self.blocked_users:
            await self._send_blocked_message(interaction)
            return

        if not await self._is_member_of_access_guild(interaction):
            await self._send_access_guild_required_message(interaction)
            return

        tier = await self._determine_user_tier(interaction)
        active_contexts = self._get_active_contexts(user_id)
        global_active = len(active_contexts) + self._get_remote_active_slots(user_id)
        if global_active >= tier.max_parallel_generations:
            await self._send_active_generation_message(
                interaction,
                max_allowed=tier.max_parallel_generations,
                active_count=global_active,
            )
            return

        logger.info(
            "gen[%s|%s] request received • type=%s workflow=%s resolution=%s",
            user_id,
            self._format_user_for_log(interaction.user),
            workflow_type,
            workflow or "default",
            resolution or "default",
        )

        is_supporter = tier.daily_limit is None
        is_donor = is_supporter or user_id in self.donor_users

        final_prompt, preset_name = self.workflow_manager.apply_prompt_preset(prompt_preset, prompt)

        context = GenerationContext(
            user_id=user_id,
            user=interaction.user,
            workflow_type=workflow_type,
            is_donor=is_donor,
            tier=tier,
            daily_limit=tier.daily_limit,
            prompt=final_prompt,
            prompt_preset_name=preset_name,
            settings=settings,
            resolution=resolution,
        )

        context.force_spoiler = self._prompt_contains_spoiler_tag(final_prompt)

        try:
            if tier.daily_limit is not None:
                current_usage = self.user_generation_counts[user_id]
                limit = tier.daily_limit
                if current_usage >= limit:
                    await self._send_limit_reached_message(interaction, limit, tier.name)
                    return

                self.user_generation_counts[user_id] = current_usage + 1
                context.counted_usage = True
                self._save_generation_counts()

                try:
                    asyncio.create_task(
                        self._publish_limit_update(
                            user_id=int(user_id),
                            used=int(self.user_generation_counts[user_id]),
                            limit=limit,
                            reset_at=float(self.last_reset_time + 86400.0),
                            tier_name=tier.name,
                            queue_priority=tier.queue_priority,
                            max_parallel=tier.max_parallel_generations,
                        )
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("SYNC publish skipped: %s", exc)

            self.active_generations[user_id].append(context)
            context.slot_counted = True
            try:
                asyncio.create_task(
                self._publish_active_delta(
                    user_id=int(user_id),
                    delta=1,
                        tier_name=tier.name,
                        queue_priority=tier.queue_priority,
                        max_parallel=tier.max_parallel_generations,
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("SYNC active publish skipped: %s", exc)
            await self._process_generation(
                interaction,
                workflow_type,
                final_prompt,
                workflow,
                settings,
                resolution,
                input_image,
                context,
            )

        except Exception as exc:
            if not interaction.response.is_done():
                await self._send_error_message(interaction, str(exc))
            logger.error("Generation error: %s", exc, exc_info=True)
            self._finalize_generation_context(context, success=False)
            raise
    async def _process_generation(
        self,
        interaction: discord.Interaction,
        workflow_type: str,
        prompt: str,
        workflow: Optional[str],
        settings: Optional[str],
        resolution: Optional[str],
        input_image: Optional[discord.Attachment],
        context: GenerationContext,
    ) -> None:
        workflow_name = workflow or self.workflow_manager.get_default_workflow(workflow_type)
        context.workflow_name = workflow_name

        workflow_config = self.workflow_manager.get_workflow(workflow_name)
        if resolution:
            context.resolution = resolution
        elif not context.resolution:
            context.resolution = workflow_config.get("default_resolution")
        security_results = await self.hook_manager.execute_hook(
            "is.security",
            interaction,
            workflow_name,
            workflow_type,
            prompt,
            workflow_config,
            settings,
        )

        for result in security_results:
            if not result.state:
                embed = ui_embeds.build_notice_embed(
                    title="❌ Security check failed",
                    description=result.message or "Generation was rejected by security policy.",
                    color=ui_embeds.ERROR_COLOR,
                )
                await interaction.response.send_message(embed=embed, ephemeral=True)
                self._finalize_generation_context(context, success=False)
                return

        if not workflow_config:
            embed = ui_embeds.build_notice_embed(
                title="❌ Workflow not found",
                description=f"Workflow `{workflow_name}` is not available. Use /workflows to list options.",
                color=ui_embeds.ERROR_COLOR,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            self._finalize_generation_context(context, success=False)
            return

        if workflow_config.get("type", "txt2img") != workflow_type:
            embed = ui_embeds.build_notice_embed(
                title="❌ Workflow type mismatch",
                description=f"Workflow `{workflow_name}` does not support `{workflow_type}`.",
                color=ui_embeds.ERROR_COLOR,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
            self._finalize_generation_context(context, success=False)
            return

        image_data: Optional[bytes] = None
        if workflow_type in ["img2img", "upscale"]:
            if not input_image or not input_image.filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                embed = ui_embeds.build_notice_embed(
                    title="❌ Invalid image",
                    description="Provide a valid PNG/JPG/JPEG/WEBP image for this workflow.",
                    color=ui_embeds.ERROR_COLOR,
                )
                await interaction.response.send_message(embed=embed, ephemeral=True)
                self._finalize_generation_context(context, success=False)
                return
            image_data = await input_image.read()

        queue_position = self.generation_queue.get_queue_position()
        status = (
            f"⏳ Waiting in queue • position {queue_position + 1}"
            if queue_position > 0
            else "🚀 Preparing your generation…"
        )

        context.view = self._create_generation_view(context)
        embed = self._build_generation_embed(
            context,
            status=status,
            title="🎨 Generation queued",
            color=ui_embeds.ACCENT_COLOR,
            extra_fields=[
                ("📬 Queue position", str(queue_position + 1), True)
                if queue_position > 0
                else ("📬 Queue position", "Active", True)
            ],
        )
        await interaction.response.send_message(embed=embed, view=context.view)
        context.message = await interaction.original_response()

        await self.generation_queue.add_to_queue(
            self._run_generation_pipeline,
            context,
            workflow_name,
            workflow_type,
            prompt,
            settings,
            context.resolution,
            image_data,
            priority=context.tier.queue_priority,
        )
    async def _run_generation_pipeline(
        self,
        context: GenerationContext,
        workflow_name: str,
        workflow_type: str,
        prompt: Optional[str],
        settings: Optional[str],
        resolution: Optional[str],
        image_data: Optional[bytes],
    ) -> None:
        start_ts = time.time()
        context.workflow_name = workflow_name
        context.prompt = prompt
        context.settings = settings
        if resolution:
            context.resolution = resolution

        try:
            if context.cancel_event.is_set():
                await self._handle_cancelled_generation(context)
                return

            workflow_json = self.workflow_manager.prepare_workflow(
                workflow_name,
                prompt,
                settings,
                context.resolution,
                image_data,
            )

            if context.cancel_event.is_set():
                await self._handle_cancelled_generation(context)
                return

            logger.info(
                "gen[%s|%s] submitting workflow=%s type=%s",
                context.user_id,
                self._format_user_for_log(context.user),
                workflow_name,
                workflow_type,
            )

            if not self.comfy_client:
                raise RuntimeError("ComfyUI client is not available")

            result = await self.comfy_client.generate(workflow_json)
            if "error" in result:
                raise RuntimeError(result["error"])

            prompt_id = result.get("prompt_id")
            if not prompt_id:
                raise RuntimeError("No prompt ID received from ComfyUI")

            context.prompt_id = prompt_id

            async def update(status: str, image_file: Optional[discord.File] = None) -> None:
                color, title = self._determine_status_style(status, image_file is not None)
                elapsed = time.time() - start_ts
                await self._update_generation_message(
                    context,
                    status=status,
                    title=title,
                    color=color,
                    extra_fields=[("⏱ Elapsed", f"{elapsed:.1f}s", True)],
                    image_file=image_file,
                )

            await update("🚀 Prompt submitted to ComfyUI")
            await self.comfy_client.listen_for_updates(
                prompt_id,
                update,
                cancel_event=context.cancel_event,
            )

            if context.cancel_event.is_set():
                await self._handle_cancelled_generation(context)
                return

            context.completed = True
            logger.info(
                "gen[%s|%s] completed workflow=%s in %.2fs",
                context.user_id,
                self._format_user_for_log(context.user),
                workflow_name,
                time.time() - start_ts,
            )

        except Exception as exc:
            logger.error(
                "gen[%s|%s] failed: %s",
                context.user_id,
                self._format_user_for_log(context.user),
                exc,
                exc_info=True,
            )
            await self._update_generation_message(
                context,
                status=f"❌ {exc}",
                title="❌ Generation failed",
                color=ui_embeds.ERROR_COLOR,
            )
        finally:
            if context.view:
                context.view.disable()
            if context.message:
                try:
                    await context.message.edit(view=context.view)
                except discord.HTTPException:
                    pass
            self._finalize_generation_context(context, success=context.completed)
    def _create_generation_view(self, context: GenerationContext) -> GenerationView:
        async def on_cancel(interaction: discord.Interaction) -> None:
            await self._handle_cancel_request(context, interaction)

        return GenerationView(context.user.id, on_cancel)

    def _determine_status_style(self, status: str, has_image: bool) -> tuple[int, str]:
        if has_image or status.startswith("✅") or status.startswith("🖼"):
            return ui_embeds.SUCCESS_COLOR, "✅ Generation update"
        if status.startswith("🛑"):
            return ui_embeds.WARNING_COLOR, "🛑 Generation update"
        if status.startswith("❌"):
            return ui_embeds.ERROR_COLOR, "❌ Generation update"
        return ui_embeds.PROGRESS_COLOR, "🎨 Generation update"

    async def _update_generation_message(
        self,
        context: GenerationContext,
        *,
        status: str,
        title: str,
        color: int,
        extra_fields: Optional[List[ui_embeds.EmbedField]] = None,
        image_file: Optional[discord.File] = None,
    ) -> None:
        if not context.message:
            return

        embed = self._build_generation_embed(
            context,
            status=status,
            title=title,
            color=color,
            extra_fields=extra_fields,
        )

        kwargs = {"embed": embed}
        if context.view:
            kwargs["view"] = context.view
        if image_file:
            if context.force_spoiler and not image_file.filename.startswith("SPOILER_"):
                image_file.filename = f"SPOILER_{image_file.filename}"
                try:
                    image_file.spoiler = True
                except AttributeError:
                    pass
            kwargs["attachments"] = [image_file]

        try:
            await context.message.edit(**kwargs)
        except discord.HTTPException as exc:  # pragma: no cover - defensive
            logger.debug("Failed to update message for %s: %s", context.user_id, exc)

    def _build_generation_embed(
        self,
        context: GenerationContext,
        *,
        status: str,
        title: str,
        color: int,
        extra_fields: Optional[List[ui_embeds.EmbedField]] = None,
    ) -> discord.Embed:
        fields: List[ui_embeds.EmbedField] = [("🎯 Mode", context.workflow_type.upper(), True)]
        if extra_fields:
            fields.extend(extra_fields)
        if context.prompt_preset_name:
            fields.append(("🏷️ Preset", context.prompt_preset_name, True))
        if context.resolution:
            fields.append(("🖼️ Resolution", context.resolution, True))
        fields.append(("🕒 Started", f"<t:{int(context.started_at)}:R>", True))

        return ui_embeds.build_generation_embed(
            title=title,
            user=context.user,
            workflow_name=context.workflow_name or "—",
            status=status,
            color=color,
            prompt=context.prompt,
            settings=context.settings,
            usage=self._usage_text(context),
            fields=fields,
        )

    def _usage_text(self, context: GenerationContext) -> Optional[str]:
        if context.is_donor:
            return "💎 Unlimited access"

        used = self.user_generation_counts.get(context.user_id, 0)
        if context.daily_limit is None:
            return None
        return ui_embeds.format_usage_bar(
            used,
            context.daily_limit,
            reset_hint=f"resets in {self._format_time_remaining()}",
        )

    async def _handle_cancel_request(self, context: GenerationContext, interaction: discord.Interaction) -> None:
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=True)

        if context.cancel_event.is_set():
            await interaction.followup.send("Generation already cancelled.", ephemeral=True)
            return

        logger.info(
            "gen[%s|%s] cancellation requested",
            context.user_id,
            self._format_user_for_log(context.user),
        )
        context.cancel_event.set()
        await self.generation_queue.cancel_pending(context)

        if self.comfy_client and context.prompt_id:
            try:
                await self.comfy_client.cancel_prompt(context.prompt_id)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Cancel prompt failed: %s", exc)

        await self._handle_cancelled_generation(context)
        self._finalize_generation_context(context, success=False)
        await interaction.followup.send("Generation cancelled.", ephemeral=True)

    async def _handle_cancelled_generation(self, context: GenerationContext, *, reason: str = "Generation cancelled by user.") -> None:
        if context.cancelled_notified:
            return

        context.cancelled_notified = True
        if context.view:
            context.view.disable()

        await self._update_generation_message(
            context,
            status=f"🛑 {reason}",
            title="🛑 Generation cancelled",
            color=ui_embeds.WARNING_COLOR,
        )

    def _finalize_generation_context(self, context: GenerationContext, *, success: bool) -> None:
        if context.finalized:
            return

        context.finalized = True
        active = self.active_generations.get(context.user_id, [])
        if active:
            try:
                active.remove(context)
            except ValueError:
                pass
            if active:
                self.active_generations[context.user_id] = active
            else:
                self.active_generations.pop(context.user_id, None)

        if success:
            self._record_successful_generation(context.user_id)

        if context.counted_usage and not success and not context.cancel_event.is_set():
            new_value = max(0, self.user_generation_counts.get(context.user_id, 0) - 1)
            self.user_generation_counts[context.user_id] = new_value
            self._save_generation_counts()

            try:
                asyncio.create_task(
                    self._publish_limit_update(
                        user_id=int(context.user_id),
                        used=new_value,
                        limit=context.daily_limit or self._get_public_tier().daily_limit or 0,
                        reset_at=float(self.last_reset_time + 86400.0),
                        tier_name=context.tier.name,
                        queue_priority=context.tier.queue_priority,
                        max_parallel=context.tier.max_parallel_generations,
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("SYNC publish rollback skipped: %s", exc)

        if context.slot_counted:
            try:
                asyncio.create_task(
                    self._publish_active_delta(
                        user_id=int(context.user_id),
                        delta=-1,
                        tier_name=context.tier.name,
                        queue_priority=context.tier.queue_priority,
                        max_parallel=context.tier.max_parallel_generations,
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("SYNC active rollback skipped: %s", exc)

    async def _send_blocked_message(self, interaction: discord.Interaction) -> None:
        embed = ui_embeds.build_notice_embed(
            title="🚫 Access restricted",
            description="Your account is blocked from using this bot. Contact support if this is unexpected.",
            color=ui_embeds.ERROR_COLOR,
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    async def _send_active_generation_message(
        self,
        interaction: discord.Interaction,
        *,
        max_allowed: int,
        active_count: int,
    ) -> None:
        embed = ui_embeds.build_notice_embed(
            title="⏳ Already processing",
            description=(
                "Please wait for your existing generations to finish before starting a new one.\n"
                f"Slots in use across all bots: **{active_count}** / **{max_allowed}**"
            ),
            color=ui_embeds.WARNING_COLOR,
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    async def _send_limit_reached_message(self, interaction: discord.Interaction, limit: int, tier_name: str) -> None:
        usage = ui_embeds.format_usage_bar(
            self.user_generation_counts[str(interaction.user.id)],
            limit,
            reset_hint=f"resets in {self._format_time_remaining()}",
        )
        embed = ui_embeds.build_limit_embed(
            description=(
                f"You've reached the daily limit for the {tier_name} tier.\n"
                "Support us to unlock unlimited generations!\n\n"
                f"{usage}"
            )
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    async def _send_error_message(self, interaction: discord.Interaction, error: str) -> None:
        embed = ui_embeds.build_notice_embed(
            title="❌ Unexpected error",
            description=f"```{error[:1000]}```",
            color=ui_embeds.ERROR_COLOR,
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    async def _is_member_of_access_guild(self, interaction: discord.Interaction) -> bool:
        try:
            if not self.access_guild_id:
                logger.debug("access-guild: skip (id empty)")
                return True

            member = await self._get_access_member(interaction)
            return member is not None
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("access-guild: failed %s", exc)
            return False

    async def _send_access_guild_required_message(self, interaction: discord.Interaction) -> None:
        embed = ui_embeds.build_notice_embed(
            title="🚫 Access restricted",
            description="Join the required server to start generations. https://discord.gg/XnxmanFBUp",
            color=ui_embeds.ERROR_COLOR,
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    async def _has_unlimited_access(self, interaction: discord.Interaction) -> bool:
        tier = await self._determine_user_tier(interaction)
        return tier.daily_limit is None
