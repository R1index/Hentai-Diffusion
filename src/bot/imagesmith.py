from __future__ import annotations
import asyncio
import hashlib
import hmac
import importlib
import inspect
import json
import os
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import discord
import yaml
from discord import app_commands
from discord.ext import commands, tasks

from logger import logger
from .commands import rgen_command, workflows_command
from ..comfy.client import ComfyUIClient
from ..comfy.workflow_manager import WorkflowManager
from ..core.generation_queue import GenerationQueue
from ..core.hook_manager import HookManager
from ..core.plugin import Plugin
from ..core.security import BasicSecurity, SecurityManager
from ..ui import embeds as ui_embeds
from ..ui.views import GenerationView


@dataclass
class GenerationContext:
    """State for a single in-flight generation."""

    user_id: str
    user: discord.abc.User
    workflow_type: str
    is_donor: bool
    prompt: Optional[str] = None
    settings: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    workflow_name: Optional[str] = None
    message: Optional[discord.Message] = None
    prompt_id: Optional[str] = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    view: Optional[GenerationView] = None
    counted_usage: bool = False
    completed: bool = False
    cancelled_notified: bool = False
    finalized: bool = False


class ComfyUIBot(commands.Bot):
    GENERATION_COUNTS_FILE = "generation_counts.yml"
    DAILY_GENERATION_LIMIT = 50
    QUEUE_STUCK_THRESHOLD = 1800  # 30 minutes

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
        self.active_generations: Dict[str, GenerationContext] = {}

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
        self.user_generation_counts: Dict[str, int] = defaultdict(int)
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

    def _load_security_lists(self) -> None:
        try:
            config_data = None
            for encoding in ("utf-8", "utf-8-sig", "cp1251"):
                try:
                    with open(self.configuration_path, "r", encoding=encoding) as file:
                        config_data = yaml.safe_load(file) or {}
                    break
                except UnicodeDecodeError:
                    continue

            if config_data is None:
                raise UnicodeDecodeError("auto", b"", 0, 1, "Unable to decode config in known encodings")

            security = config_data.get("security", {}) or {}
            self.blocked_users = {str(user) for user in security.get("blocked_users", [])}
            self.donor_users = {str(user) for user in security.get("donor_users", [])}

            access_guild_id = security.get("access_guild_id")
            self.access_guild_id = str(access_guild_id) if access_guild_id else None
            self.supporter_role_name = security.get("supporter_role_name", "Supporter")

            supporter_role_id = security.get("supporter_role_id")
            self.supporter_role_id = str(supporter_role_id) if supporter_role_id else None

            logger.info(
                "Security configuration loaded • blocked=%d donors=%d",
                len(self.blocked_users),
                len(self.donor_users),
            )

            if self.access_guild_id:
                role_descriptor = self.supporter_role_id or self.supporter_role_name
                logger.info(
                    "Supporter role checks enabled • guild=%s role=%s",
                    self.access_guild_id,
                    role_descriptor,
                )
            else:
                logger.warning("Supporter role checks disabled: access_guild_id is not configured")

        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to load security configuration: %s", exc)
            self.blocked_users = set()
            self.donor_users = set()
            self.access_guild_id = None
            self.supporter_role_name = "Supporter"
            self.supporter_role_id = None

    def _load_generation_counts(self) -> None:
        counts_path = self._get_counts_path()
        try:
            if os.path.exists(counts_path):
                with open(counts_path, "r", encoding="utf-8") as file:
                    data = yaml.safe_load(file) or {}
                    self.user_generation_counts = defaultdict(int, data.get("counts", {}))
                    self.last_reset_time = data.get("last_reset", time.time())
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

    def _format_time_remaining(self) -> str:
        remaining = max(0.0, 86400 - (time.time() - self.last_reset_time))
        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)
        return f"{hours}h {minutes}m"

    # ------------------------------------------------------------------
    # Background monitoring
    # ------------------------------------------------------------------
    @tasks.loop(minutes=5)
    async def monitor_generations(self) -> None:
        """Cancel generations that appear to be stuck."""

        current_time = time.time()
        stuck_contexts = [
            context
            for context in list(self.active_generations.values())
            if current_time - context.started_at > self.QUEUE_STUCK_THRESHOLD
        ]

        for context in stuck_contexts:
            logger.warning("gen[%s] timed out — cancelling", context.user_id)
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

            self._sync_prune_seen()
            if event_id in self._sync_seen:
                return
            self._sync_seen[event_id] = self._sync_now() + self._sync_seen_ttl

            if self.user and src_bot == self.user.id:
                return

            user_id = str(int(payload["user_id"]))
            used = int(payload["generations_used"])
            limit = int(payload["limit"])
            reset_at = float(payload["reset_at"])
            incoming_last_reset = reset_at - 86400.0

            if getattr(self, "last_reset_time", 0) > incoming_last_reset + 2:
                logger.debug("SYNC skipped — local reset is newer")
                return

            self.user_generation_counts[user_id] = used
            self.last_reset_time = incoming_last_reset
            self._save_generation_counts()

            logger.info(
                "SYNC applied from bot %s → user %s: %s/%s",
                src_bot,
                user_id,
                used,
                limit,
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

    async def _publish_limit_update(self, user_id: int, used: int, limit: int, reset_at: float) -> None:
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
                "event_id": event_id,
                "ts": int(self._sync_now()),
                "source_bot_id": self.user.id if self.user else 0,
                "user_id": int(user_id),
                "generations_used": int(used),
                "limit": int(limit),
                "reset_at": float(reset_at),
            }
            wrapper = {"payload": payload, "sig": self._sync_sign(payload)}
            await channel.send(self.SYNC_PREFIX + self._sync_canon(wrapper))
            self._sync_seen[event_id] = self._sync_now() + self._sync_seen_ttl
            self._sync_prune_seen()
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("publish_limit_update failed: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------
    def _create_limits_command(self):
        @app_commands.command(name="limits", description="Check your current generation limits and status")
        async def limits_command(interaction: discord.Interaction) -> None:
            user_id = str(interaction.user.id)
            self._reset_counts_if_needed()

            is_donor = user_id in self.donor_users or await self._has_unlimited_access(interaction)
            count = self.user_generation_counts.get(user_id, 0)

            if is_donor:
                description = (
                    "🌟 **Donor status**: unlimited access!\n"
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
                    self.DAILY_GENERATION_LIMIT,
                    reset_hint=f"resets in {self._format_time_remaining()}",
                )
                description = (
                    "🔒 You are using the public tier.\n"
                    "Support us to unlock unlimited generations!"
                )
                embed = ui_embeds.build_notice_embed(
                    title="📊 Daily usage",
                    description=f"{description}\n\n{usage}",
                    color=ui_embeds.WARNING_COLOR,
                )

            await interaction.response.send_message(embed=embed, ephemeral=True)

        return limits_command
    async def handle_generation(
        self,
        interaction: discord.Interaction,
        workflow_type: str,
        prompt: str,
        workflow: Optional[str] = None,
        settings: Optional[str] = None,
        input_image: Optional[discord.Attachment] = None,
    ) -> None:
        user_id = str(interaction.user.id)
        self._reset_counts_if_needed()

        if user_id in self.blocked_users:
            await self._send_blocked_message(interaction)
            return

        if user_id in self.active_generations and not self.active_generations[user_id].finalized:
            await self._send_active_generation_message(interaction)
            return

        logger.info(
            "gen[%s] request received • type=%s workflow=%s",
            user_id,
            workflow_type,
            workflow or "default",
        )

        is_supporter = await self._has_unlimited_access(interaction)
        is_donor = is_supporter or user_id in self.donor_users

        context = GenerationContext(
            user_id=user_id,
            user=interaction.user,
            workflow_type=workflow_type,
            is_donor=is_donor,
            prompt=prompt,
            settings=settings,
        )

        try:
            if not is_donor:
                current_usage = self.user_generation_counts[user_id]
                if current_usage >= self.DAILY_GENERATION_LIMIT:
                    await self._send_limit_reached_message(interaction)
                    return

                self.user_generation_counts[user_id] = current_usage + 1
                context.counted_usage = True
                self._save_generation_counts()

                try:
                    asyncio.create_task(
                        self._publish_limit_update(
                            user_id=int(user_id),
                            used=int(self.user_generation_counts[user_id]),
                            limit=self.DAILY_GENERATION_LIMIT,
                            reset_at=float(self.last_reset_time + 86400.0),
                        )
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("SYNC publish skipped: %s", exc)

            self.active_generations[user_id] = context
            await self._process_generation(
                interaction,
                workflow_type,
                prompt,
                workflow,
                settings,
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
        input_image: Optional[discord.Attachment],
        context: GenerationContext,
    ) -> None:
        workflow_name = workflow or self.workflow_manager.get_default_workflow(workflow_type)
        context.workflow_name = workflow_name

        workflow_config = self.workflow_manager.get_workflow(workflow_name)
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
            image_data,
        )
    async def _run_generation_pipeline(
        self,
        context: GenerationContext,
        workflow_name: str,
        workflow_type: str,
        prompt: Optional[str],
        settings: Optional[str],
        image_data: Optional[bytes],
    ) -> None:
        start_ts = time.time()
        context.workflow_name = workflow_name
        context.prompt = prompt
        context.settings = settings

        try:
            if context.cancel_event.is_set():
                await self._handle_cancelled_generation(context)
                return

            workflow_json = self.workflow_manager.prepare_workflow(
                workflow_name,
                prompt,
                settings,
                image_data,
            )

            if context.cancel_event.is_set():
                await self._handle_cancelled_generation(context)
                return

            logger.info(
                "gen[%s] submitting workflow=%s type=%s",
                context.user_id,
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
                "gen[%s] completed workflow=%s in %.2fs",
                context.user_id,
                workflow_name,
                time.time() - start_ts,
            )

        except Exception as exc:
            logger.error("gen[%s] failed: %s", context.user_id, exc, exc_info=True)
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
        return ui_embeds.format_usage_bar(
            used,
            self.DAILY_GENERATION_LIMIT,
            reset_hint=f"resets in {self._format_time_remaining()}",
        )

    async def _handle_cancel_request(self, context: GenerationContext, interaction: discord.Interaction) -> None:
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=True)

        if context.cancel_event.is_set():
            await interaction.followup.send("Generation already cancelled.", ephemeral=True)
            return

        logger.info("gen[%s] cancellation requested", context.user_id)
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

    async def _handle_cancelled_generation(
        self,
        context: GenerationContext,
        *,
        reason: str = "Generation cancelled by user.",
    ) -> None:
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
        self.active_generations.pop(context.user_id, None)

        if context.counted_usage and not success:
            new_value = max(0, self.user_generation_counts.get(context.user_id, 0) - 1)
            self.user_generation_counts[context.user_id] = new_value
            self._save_generation_counts()

            try:
                asyncio.create_task(
                    self._publish_limit_update(
                        user_id=int(context.user_id),
                        used=new_value,
                        limit=self.DAILY_GENERATION_LIMIT,
                        reset_at=float(self.last_reset_time + 86400.0),
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("SYNC publish rollback skipped: %s", exc)

    async def _send_blocked_message(self, interaction: discord.Interaction) -> None:
        embed = ui_embeds.build_notice_embed(
            title="🚫 Access restricted",
            description="Your account is blocked from using this bot. Contact support if this is unexpected.",
            color=ui_embeds.ERROR_COLOR,
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    async def _send_active_generation_message(self, interaction: discord.Interaction) -> None:
        embed = ui_embeds.build_notice_embed(
            title="⏳ Already processing",
            description="Please wait for your current generation to finish before starting a new one.",
            color=ui_embeds.WARNING_COLOR,
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    async def _send_limit_reached_message(self, interaction: discord.Interaction) -> None:
        usage = ui_embeds.format_usage_bar(
            self.user_generation_counts[str(interaction.user.id)],
            self.DAILY_GENERATION_LIMIT,
            reset_hint=f"resets in {self._format_time_remaining()}",
        )
        embed = ui_embeds.build_limit_embed(
            description=(
                "You've reached the daily limit for the public tier.\n"
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

    async def _has_unlimited_access(self, interaction: discord.Interaction) -> bool:
        try:
            if not self.access_guild_id:
                logger.info("Supporter check: access_guild_id is empty")
                return False

            try:
                gid = int(self.access_guild_id)
            except Exception:
                logger.warning("Supporter check: invalid access_guild_id=%r", self.access_guild_id)
                return False

            if interaction.guild and interaction.guild.id == gid:
                target_guild = interaction.guild
                logger.debug("Supporter check: using interaction guild %s", gid)
            else:
                target_guild = self.get_guild(gid)
                if target_guild is None:
                    try:
                        target_guild = await self.fetch_guild(gid)
                        logger.debug("Supporter check: fetched guild %s", gid)
                    except Exception as exc:
                        logger.warning("Supporter check: fetch_guild(%s) failed: %s", gid, exc)
                        return False

            member = target_guild.get_member(interaction.user.id)
            if member is None:
                try:
                    member = await target_guild.fetch_member(interaction.user.id)
                    logger.debug("Supporter check: fetched member %s on guild %s", interaction.user.id, gid)
                except Exception as exc:
                    logger.info("Supporter check: user %s is not in guild %s: %s", interaction.user.id, gid, exc)
                    return False

            role = None
            if getattr(self, "supporter_role_id", None):
                try:
                    rid = int(self.supporter_role_id)
                    role = target_guild.get_role(rid)
                    if role:
                        logger.debug("Supporter check: found role by ID %s: %s", rid, role.name)
                except Exception:
                    role = None

            if role is None and getattr(self, "supporter_role_name", None):
                lname = self.supporter_role_name.casefold()
                for candidate in target_guild.roles:
                    if candidate.name.casefold() == lname:
                        role = candidate
                        logger.debug("Supporter check: found role by name %s -> %s", self.supporter_role_name, candidate.id)
                        break

            if not role:
                logger.warning(
                    "Supporter check: role not found on guild (guild_id=%s, role_id=%s, role_name=%r)",
                    gid,
                    getattr(self, "supporter_role_id", None),
                    getattr(self, "supporter_role_name", None),
                )
                return False

            member_role_ids = {r.id for r in getattr(member, "roles", [])}
            has_role = role.id in member_role_ids
            logger.info(
                "Supporter check: guild=%s member=%s role=%s/%s has=%s",
                gid,
                member.id,
                role.id,
                role.name,
                has_role,
            )
            return has_role

        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Supporter role check failed: %s", exc, exc_info=True)
            return False
