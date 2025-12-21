from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import sys
from pathlib import Path
from typing import List
import time

import discord
from discord import app_commands
from discord.ext import commands, tasks

from logger import logger
from src.app import ConfigManager, GenerationOrchestrator, GenerationStore, SyncBridge
from src.app.sync import SyncConfig
from src.core.generation_queue import GenerationQueue
from src.core.hook_manager import HookManager
from src.core.plugin import Plugin
from src.core.security import SecurityManager
from src.comfy.workflow_manager import WorkflowManager
from src.ui import embeds as ui_embeds
from src.bot import commands as command_factory


class DiscordBot(commands.Bot):
    """Discord-facing bot that wires commands to the GenerationOrchestrator."""

    GENERATION_COUNTS_FILE = "data/generation_counts.yml"

    def __init__(self, config_path: str = "configuration.yml", plugins_path: str = "plugins"):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.guilds = True
        super().__init__(command_prefix="/", intents=intents)

        self.config_manager = ConfigManager(config_path)
        self.plugins_path = plugins_path
        self.hook_manager = HookManager()
        self.security_manager = SecurityManager()
        self.generation_store = GenerationStore(Path(self.GENERATION_COUNTS_FILE))
        self.queue = GenerationQueue()
        self.workflow_manager = WorkflowManager(config_path)
        self.sync_bridge = SyncBridge(self, self.config_manager.sync_config())
        self.orchestrator = GenerationOrchestrator(
            config=self.config_manager,
            hook_manager=self.hook_manager,
            security_manager=self.security_manager,
            generation_store=self.generation_store,
            queue=self.queue,
            sync_bridge=self.sync_bridge,
            workflow_manager=self.workflow_manager,
        )
        self.plugins: List[Plugin] = []
        self.monitor_generations.start()

    # ---------------------- Lifecycle ----------------------
    async def setup_hook(self) -> None:
        logger.info("Setting up bot…")
        await self._load_plugins()
        try:
            await self.orchestrator.setup_comfy()
        except Exception:
            await self.close()
            sys.exit(1)

        logger.info("Registering slash commands")
        try:
            self.tree.add_command(command_factory.rgen_command(self))
            self.tree.add_command(command_factory.workflows_command(self))
            self.tree.add_command(self._create_limits_command())
            self.tree.add_command(self._create_spoiler_command())
            self.tree.add_command(command_factory.profile_command(self))
            # legacy commands
            self.tree.add_command(command_factory.reforge_command(self))
            self.tree.add_command(command_factory.upscale_command(self))

            synced = await self.tree.sync()
            logger.info("Slash commands synced: %s", ", ".join(f"/{cmd.name}" for cmd in synced) or "none")
        except Exception as exc:
            logger.error("Failed to sync commands: %s", exc)
            await self.close()
            sys.exit(1)

    async def on_ready(self) -> None:
        logger.info("Logged in as %s (ID: %s)", self.user, self.user.id if self.user else "?")
        logger.info("Connected to %d guilds", len(self.guilds))
        perms = discord.Permissions(
            send_messages=True,
            read_messages=True,
            attach_files=True,
            embed_links=True,
            use_external_emojis=True,
            add_reactions=True,
            read_message_history=True,
        )
        invite = discord.utils.oauth_url(self.user.id, permissions=perms, scopes=("bot", "applications.commands"))
        logger.info("Invite link: %s", invite)
        logger.info("Bot is ready")

    async def close(self) -> None:  # type: ignore[override]
        self.monitor_generations.cancel()
        await self.orchestrator.close()
        await super().close()

    # ---------------------- Sync channel listener ----------------------
    async def on_message(self, message: discord.Message) -> None:
        if not message.author.bot:
            return
        cfg: SyncConfig = self.config_manager.sync_config()
        if message.channel.id != cfg.channel_id or not message.content.startswith(cfg.prefix):
            return
        try:
            data = message.content[len(cfg.prefix) :]
            import json as _json

            wrapper = _json.loads(data)
            parsed = self.sync_bridge.handle_sync_payload(wrapper, source_bot_id=message.author.id)
            if not parsed:
                return
            if parsed["kind"] == "limit":
                payload = parsed["payload"]
                user_id = str(int(payload["user_id"]))
                used = int(payload["generations_used"])
                reset_at = float(payload["reset_at"])
                limit_raw = payload.get("tier", {}).get("limit", payload.get("limit"))
                limit = None if limit_raw in (None, -1) else int(limit_raw)
                incoming_last_reset = reset_at - 86400.0
                if self.generation_store.last_reset > incoming_last_reset + 2:
                    return
                if limit is None:
                    return
                self.generation_store.set_count(user_id, used)
                self.generation_store.last_reset = incoming_last_reset
                self.generation_store._save()
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("SYNC parse/apply failed: %s", exc, exc_info=True)

    # ---------------------- Background tasks ----------------------
    @tasks.loop(minutes=5)
    async def monitor_generations(self) -> None:
        current_time = time.time()
        stuck_contexts = [
            context
            for contexts in list(self.orchestrator.active_generations.values())
            for context in contexts
            if current_time - context.started_at > self.orchestrator.QUEUE_STUCK_THRESHOLD
        ]
        for context in stuck_contexts:
            logger.warning("gen[%s|%s] timed out — cancelling", context.user_id, context.user)
            context.cancel_event.set()
            if self.orchestrator.comfy_client and context.prompt_id:
                try:
                    await self.orchestrator.comfy_client.cancel_prompt(context.prompt_id)
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("Timeout cancellation failed: %s", exc)
            await self.orchestrator._handle_cancelled_generation(context, reason="Generation timed out.")
            self.orchestrator._finalize_generation_context(context, success=False)

    @monitor_generations.before_loop
    async def before_monitor(self) -> None:
        await self.wait_until_ready()

    # ---------------------- Commands (limits, spoiler) ----------------------
    def _create_limits_command(self):
        @app_commands.command(name="limits", description="Check your current generation limits and status")
        async def limits(interaction: discord.Interaction) -> None:
            user_id = str(interaction.user.id)
            self.generation_store.reset_daily_if_needed()
            tier = await self.orchestrator.determine_user_tier(interaction)
            is_donor = tier.daily_limit is None
            count = self.generation_store.get_count(user_id)
            queue_info = f"Queue slots: **{tier.max_parallel_generations}** at once."
            priority_hint = "Priority over lower tiers." if tier.queue_priority >= 30 else "Standard queue priority."
            if is_donor:
                description = (
                    f"🌟 **{tier.name}**: unlimited access!\n{queue_info}\n{priority_hint}\nThank you for supporting the project."
                )
                embed = ui_embeds.build_notice_embed(
                    title="💎 Unlimited access",
                    description=description,
                    color=ui_embeds.SUCCESS_COLOR,
                )
            else:
                usage = ui_embeds.format_usage_bar(
                    count, tier.daily_limit or 0, reset_hint="resets in 24h"
                )
                description = (
                    f"🔒 You are using the {tier.name} tier.\n{queue_info}\n{priority_hint}\nSupport us to unlock unlimited generations!"
                )
                embed = ui_embeds.build_notice_embed(
                    title="📊 Daily usage", description=f"{description}\n\n{usage}", color=ui_embeds.WARNING_COLOR
                )
            await interaction.response.send_message(embed=embed, ephemeral=True)

        return limits

    def _create_spoiler_command(self) -> app_commands.Group:
        spoiler_group = app_commands.Group(name="spoiler", description="Manage spoiler tags")

        @spoiler_group.command(name="list", description="Show configured spoiler tags")
        async def spoiler_list(interaction: discord.Interaction) -> None:
            tags = self.orchestrator.spoiler_tags()
            if tags:
                description = "\n".join(f"• `{tag}`" for tag in tags)
                title = "📑 Spoiler tags"
                color = ui_embeds.ACCENT_COLOR
            else:
                description = "No spoiler tags are configured. Use /spoiler add to create one."
                title = "ℹ️ No spoiler tags"
                color = ui_embeds.WARNING_COLOR
            embed = ui_embeds.build_notice_embed(title=title, description=description, color=color)
            await interaction.response.send_message(embed=embed, ephemeral=True)

        @spoiler_group.command(name="add", description="Add a tag that will force image spoilers")
        @app_commands.describe(tag="Tag to treat as a spoiler trigger")
        async def spoiler_add(interaction: discord.Interaction, tag: str) -> None:
            perms = getattr(interaction.user, "guild_permissions", None)
            if not (perms and perms.manage_guild):
                embed = ui_embeds.build_limit_embed("You need the **Manage Server** permission to modify spoiler tags.")
                await interaction.response.send_message(embed=embed, ephemeral=True)
                return
            cleaned = tag.strip()
            if not cleaned:
                embed = ui_embeds.build_notice_embed(
                    title="❌ Invalid tag", description="Provide a non-empty tag to add.", color=ui_embeds.ERROR_COLOR
                )
                await interaction.response.send_message(embed=embed, ephemeral=True)
                return
            normalized = self.config_manager.normalize_tag(cleaned)
            current_norm = {self.config_manager.normalize_tag(t) for t in self.orchestrator.spoiler_tags()}
            if normalized in current_norm:
                embed = ui_embeds.build_notice_embed(
                    title="ℹ️ Tag already exists",
                    description=f"`{cleaned}` is already configured as a spoiler tag.",
                    color=ui_embeds.WARNING_COLOR,
                )
                await interaction.response.send_message(embed=embed, ephemeral=True)
                return
            tags = self.orchestrator.spoiler_tags() + [cleaned]
            self.orchestrator.update_spoiler_tags(tags)
            embed = ui_embeds.build_notice_embed(
                title="✅ Spoiler tag added",
                description=f"Images will now be hidden behind spoilers when prompts include `{cleaned}`.",
                color=ui_embeds.SUCCESS_COLOR,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)

        @spoiler_group.command(name="remove", description="Remove a configured spoiler tag")
        @app_commands.describe(tag="Tag to remove from the spoiler list")
        async def spoiler_remove(interaction: discord.Interaction, tag: str) -> None:
            perms = getattr(interaction.user, "guild_permissions", None)
            if not (perms and perms.manage_guild):
                embed = ui_embeds.build_limit_embed("You need the **Manage Server** permission to modify spoiler tags.")
                await interaction.response.send_message(embed=embed, ephemeral=True)
                return
            normalized = self.config_manager.normalize_tag(tag)
            tags = self.orchestrator.spoiler_tags()
            norm_map = {self.config_manager.normalize_tag(t): t for t in tags}
            if normalized not in norm_map:
                embed = ui_embeds.build_notice_embed(
                    title="❌ Tag not found",
                    description=f"`{tag}` is not configured as a spoiler tag.",
                    color=ui_embeds.ERROR_COLOR,
                )
                await interaction.response.send_message(embed=embed, ephemeral=True)
                return
            removed_display = norm_map[normalized]
            remaining = [t for t in tags if t != removed_display]
            self.orchestrator.update_spoiler_tags(remaining)
            embed = ui_embeds.build_notice_embed(
                title="🗑️ Spoiler tag removed",
                description=f"`{removed_display}` will no longer force spoilered images.",
                color=ui_embeds.SUCCESS_COLOR,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)

        return spoiler_group

    # ---------------------- Plugin loader ----------------------
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
                spec.loader.exec_module(module)  # type: ignore[arg-type]
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
