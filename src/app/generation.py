from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, Iterable, List, Optional

import discord

from logger import logger
from src.app.config import ConfigManager
from src.app.models import GenerationContext, RoleTier
from src.app.sync import SyncBridge
from src.core.generation_queue import GenerationQueue
from src.core.hook_manager import HookManager
from src.core.security import BasicSecurity, SecurityManager
from src.ui import embeds as ui_embeds
from src.ui.views import GenerationView
from src.comfy.client import ComfyUIClient
from src.comfy.workflow_manager import WorkflowManager


class GenerationOrchestrator:
    """Coordinates Discord requests, queueing, ComfyUI calls, and sync."""

    QUEUE_STUCK_THRESHOLD = 1800  # 30 minutes
    STATS_RETENTION_DAYS = 90

    def __init__(
        self,
        *,
        config: ConfigManager,
        hook_manager: HookManager,
        security_manager: SecurityManager,
        generation_store: GenerationStore,
        queue: GenerationQueue,
        sync_bridge: SyncBridge,
        workflow_manager: WorkflowManager,
    ) -> None:
        self.config = config
        self.hook_manager = hook_manager
        self.security_manager = security_manager
        self.generation_store = generation_store
        self.queue = queue
        self.queue.set_update_callback(self._on_queue_updated)
        self.sync_bridge = sync_bridge
        self.workflow_manager = workflow_manager
        self.basic_security = BasicSecurity(self)  # registers hook
        self.comfy_client: Optional[ComfyUIClient] = None

        self.active_generations: Dict[str, List[GenerationContext]] = {}
        self._last_requests: Dict[str, Dict[str, Any]] = {}
        self._sync_channel_cache: Optional[discord.abc.Messageable] = None

        self.blocked_users = set(self.config.security_blocked())
        self.donor_users = set(self.config.security_donors())
        self.access_guild_id = self.config.access_guild_id()
        self.supporter_role_name, self.supporter_role_id = self.config.supporter_role()

        self._spoiler_tags: set[str] = set()
        self._spoiler_tag_display: Dict[str, str] = {}
        self._load_spoiler_tags(self.config.get_spoiler_tags())

        self.role_tiers: List[RoleTier] = [
            RoleTier(4, "Level 4", 1451769149045997588, None, 40, 30),
            RoleTier(3, "Level 3", 1451768900453925030, None, 30, 3),
            RoleTier(2, "Level 2", 1361296590777745560, None, 0, 1),
            RoleTier(1, "Level 1", 1450781064418299914, 100, 0, 1),
            RoleTier(0, "Public", None, 25, 0, 1),
        ]
        self.donor_tier = RoleTier(2, "Donor", None, None, 0, 1)

    # ---------------------- Lifecycle ----------------------
    async def setup_comfy(self) -> None:
        try:
            await self.hook_manager.execute_hook(
                "is.comfyui.client.before_create", self.config.comfy_instances()
            )
            self.comfy_client = ComfyUIClient(self.config.comfy_instances(), self.hook_manager)
            await self.hook_manager.execute_hook(
                "is.comfyui.client.after_create", self.config.comfy_instances()
            )
            await self.comfy_client.connect()
            logger.info("Connected to ComfyUI")
        except Exception as exc:
            logger.error("Failed to connect to ComfyUI: %s", exc)
            raise

    async def close(self) -> None:
        if self.comfy_client:
            await self.comfy_client.close()
        self.generation_store._save()

    # ---------------------- Spoilers ----------------------
    def _load_spoiler_tags(self, tags: Iterable[str]) -> None:
        self._spoiler_tags.clear()
        self._spoiler_tag_display.clear()
        for tag in tags:
            tag_str = str(tag).strip()
            if not tag_str:
                continue
            normalized = self.config.normalize_tag(tag_str)
            self._spoiler_tags.add(normalized)
            self._spoiler_tag_display[normalized] = tag_str

    def spoiler_tags(self) -> List[str]:
        return sorted(self._spoiler_tag_display.values(), key=str.lower)

    def update_spoiler_tags(self, tags: Iterable[str]) -> None:
        self._load_spoiler_tags(tags)
        self.config.set_spoiler_tags(tags)

    def prompt_contains_spoiler(self, prompt: Optional[str]) -> bool:
        if not prompt or not self._spoiler_tags:
            return False
        normalized_prompt = json.dumps(prompt, ensure_ascii=False).lower()
        return any(tag in normalized_prompt for tag in self._spoiler_tags)

    # ---------------------- Sync helpers ----------------------
    async def _get_sync_channel(self, bot: discord.Client) -> Optional[discord.abc.Messageable]:
        if self._sync_channel_cache:
            return self._sync_channel_cache
        channel = bot.get_channel(self.sync_bridge.config.channel_id)
        if not channel:
            try:
                channel = await bot.fetch_channel(self.sync_bridge.config.channel_id)
            except Exception:
                channel = None
        if channel:
            self._sync_channel_cache = channel
        return channel

    def remote_active_slots(self, user_id: str) -> int:
        return self.sync_bridge.remote_active(user_id)

    # ---------------------- Tier helpers ----------------------
    def _public_tier(self) -> RoleTier:
        return next((t for t in self.role_tiers if t.level == 0), RoleTier(0, "Public", None, 25, 0, 1))

    async def determine_user_tier(self, interaction: discord.Interaction) -> RoleTier:
        highest = self._public_tier()
        member = await self._get_access_member(interaction)
        role_ids: set[int] = {r.id for r in getattr(member, "roles", [])} if member else set()
        for tier in self.role_tiers:
            if tier.role_id and tier.role_id in role_ids and tier.level > highest.level:
                highest = tier
        if str(interaction.user.id) in self.donor_users and highest.level < self.donor_tier.level:
            highest = self.donor_tier
        return highest

    async def _get_access_guild(self, bot: discord.Client) -> Optional[discord.Guild]:
        if not self.access_guild_id:
            return None
        try:
            gid = int(self.access_guild_id)
        except Exception:
            logger.warning("access-guild: invalid id=%r", self.access_guild_id)
            return None
        guild = bot.get_guild(gid)
        if guild:
            return guild
        try:
            return await bot.fetch_guild(gid)
        except Exception as exc:
            logger.warning("access-guild: fetch_guild(%s) failed: %s", gid, exc)
            return None

    async def _get_access_member(self, interaction: discord.Interaction) -> Optional[discord.Member]:
        try:
            guild = await self._get_access_guild(interaction.client)
            if not guild:
                return None
            target = interaction.guild if interaction.guild and interaction.guild.id == guild.id else guild
            member = target.get_member(interaction.user.id)
            if member:
                return member
            return await target.fetch_member(interaction.user.id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("access-member: failed %s", exc)
            return None

    async def is_member_of_access_guild(self, interaction: discord.Interaction) -> bool:
        try:
            if not self.access_guild_id:
                return True
            return await self._get_access_member(interaction) is not None
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("access-guild: failed %s", exc)
            return False

    # ---------------------- Queue updates ----------------------
    async def _on_queue_updated(self) -> None:
        await self._refresh_queue_views()

    async def _refresh_queue_views(self) -> None:
        pending = self.queue.get_pending_contexts()
        current = self.queue.current_context
        total = len(pending) + (1 if current else 0)
        offset = 1 if current else 0
        total_display = max(total, 1)
        for idx, ctx in enumerate(pending):
            if ctx.finalized or ctx.cancel_event.is_set() or ctx.processing:
                continue
            if not ctx.message:
                continue
            position = idx + 1 + offset
            status = "⏳ Waiting in queue"
            extra_fields = [
                ("📬 Queue position", f"{position}/{total_display}", True),
                ("👥 In queue", str(total), True),
            ]
            await self._update_generation_message(
                ctx,
                status=status,
                title="🎨 Generation queued",
                color=ui_embeds.ACCENT_COLOR,
                extra_fields=extra_fields,
            )

    # ---------------------- Generation entrypoint ----------------------
    async def handle_generation(
        self,
        interaction: discord.Interaction,
        workflow_type: str,
        prompt: str,
        *,
        workflow: Optional[str] = None,
        settings: Optional[str] = None,
        resolution: Optional[str] = None,
        prompt_preset: Optional[str] = None,
        model_preset: Optional[str] = None,
        lora_preset: Optional[str] = None,
        seed: Optional[int] = None,
        input_image: Optional[discord.Attachment] = None,
    ) -> None:
        user_id = str(interaction.user.id)
        self.generation_store.reset_daily_if_needed()

        if user_id in self.blocked_users:
            await self._send_blocked_message(interaction)
            return
        if not await self.is_member_of_access_guild(interaction):
            await self._send_access_guild_required_message(interaction)
            return

        tier = await self.determine_user_tier(interaction)
        active_contexts = [ctx for ctx in self.active_generations.get(user_id, []) if not ctx.finalized]
        global_active = len(active_contexts) + self.remote_active_slots(user_id)
        if global_active >= tier.max_parallel_generations:
            await self._send_active_generation_message(
                interaction, max_allowed=tier.max_parallel_generations, active_count=global_active
            )
            return

        final_prompt, preset_name, preset_tags = self.workflow_manager.apply_prompt_preset(prompt_preset, prompt)
        model_name, model_preset_name = self.workflow_manager.apply_model_preset(model_preset)
        lora_name, lora_preset_name = self.workflow_manager.apply_lora_preset(lora_preset)

        config_params = []
        if model_name:
            config_params.append(f"model={model_name}")
        if seed is not None:
            config_params.append(f"seed={seed}")
        if lora_name:
            config_params.append(f"lora={lora_name}")

        settings_with_presets = settings
        if config_params:
            config_string = f"config({', '.join(config_params)})"
            settings_with_presets = f"{settings};{config_string}" if settings else config_string

        context = GenerationContext(
            user_id=user_id,
            user=interaction.user,
            workflow_type=workflow_type,
            is_donor=tier.daily_limit is None or user_id in self.donor_users,
            tier=tier,
            daily_limit=tier.daily_limit,
            prompt=final_prompt,
            prompt_preset_name=preset_name,
            prompt_preset_tags=preset_tags,
            model_preset_name=model_preset_name,
            lora_preset_name=lora_preset_name,
            settings=settings_with_presets,
            resolution=resolution,
            seed=seed,
        )
        context.force_spoiler = self.prompt_contains_spoiler(final_prompt)

        try:
            if tier.daily_limit is not None:
                current_usage = self.generation_store.get_count(user_id)
                if current_usage >= tier.daily_limit:
                    await self._send_limit_reached_message(interaction, tier.daily_limit, tier.name)
                    return
                self.generation_store.increment(user_id)
                context.counted_usage = True

                try:
                    channel = await self._get_sync_channel(interaction.client)
                    if channel:
                        await self.sync_bridge.publish_limit(
                            channel=channel,
                            user_id=int(user_id),
                            used=self.generation_store.get_count(user_id),
                            limit=tier.daily_limit,
                            reset_at=self.generation_store.last_reset + 86400.0,
                            tier=tier,
                        )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.debug("SYNC publish skipped: %s", exc)

            self.active_generations.setdefault(user_id, []).append(context)
            context.slot_counted = True
            try:
                channel = await self._get_sync_channel(interaction.client)
                if channel:
                    await self.sync_bridge.publish_active_delta(
                        channel=channel, user_id=int(user_id), delta=1, tier=tier
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("SYNC active publish skipped: %s", exc)

            self._store_last_request(
                user_id,
                {
                    "workflow_type": workflow_type,
                    "prompt": final_prompt,
                    "workflow": workflow,
                    "settings": settings_with_presets,
                    "resolution": resolution,
                    "prompt_preset": prompt_preset,
                    "model_preset": model_preset,
                    "lora_preset": lora_preset,
                    "seed": seed,
                },
                requires_image=input_image is not None,
            )
            await self._process_generation(
                interaction,
                workflow_type,
                final_prompt,
                workflow,
                settings_with_presets,
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

    # ---------------------- Internal processing ----------------------
    def _store_last_request(self, user_id: str, payload: Dict[str, Any], *, requires_image: bool) -> None:
        payload = dict(payload)
        payload["requires_image"] = requires_image
        self._last_requests[user_id] = payload

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
            "is.security", interaction, workflow_name, workflow_type, prompt, workflow_config, settings
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

        queue_position = self.queue.get_queue_position()
        total_queue = self.queue.size() + 1
        status = "⏳ Waiting in queue" if queue_position > 0 else "🚀 Preparing your generation…"

        context.view = self._create_generation_view(context)
        embed = self._build_generation_embed(
            context,
            status=status,
            title="🎨 Generation queued",
            color=ui_embeds.ACCENT_COLOR,
            extra_fields=[("📬 Queue position", f"{queue_position + 1}/{total_queue}", True), ("👥 In queue", str(total_queue), True)],
        )
        await interaction.response.send_message(embed=embed, view=context.view)
        context.message = await interaction.original_response()

        await self.queue.add_to_queue(
            self._run_generation_pipeline,
            context,
            workflow_name,
            workflow_type,
            prompt,
            settings,
            context.resolution,
            image_data,
            context.seed,
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
        seed: Optional[int],
    ) -> None:
        start_ts = time.time()
        context.workflow_name = workflow_name
        context.prompt = prompt
        context.settings = settings
        context.processing = True
        if context.seed is None and seed is not None:
            context.seed = seed
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
                seed=context.seed,
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
            await self.comfy_client.listen_for_updates(prompt_id, update, cancel_event=context.cancel_event)

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
                context, status=f"❌ {exc}", title="❌ Generation failed", color=ui_embeds.ERROR_COLOR
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

    # ---------------------- Helpers ----------------------
    def _determine_status_style(self, status: str, has_image: bool) -> tuple[int, str]:
        if has_image or status.startswith("✅") or status.startswith("🖼"):
            return ui_embeds.SUCCESS_COLOR, "✅ Generation update"
        if status.startswith("🛑"):
            return ui_embeds.WARNING_COLOR, "🛑 Generation update"
        if status.startswith("❌"):
            return ui_embeds.ERROR_COLOR, "❌ Generation update"
        return ui_embeds.PROGRESS_COLOR, "🎨 Generation update"

    def _create_generation_view(self, context: GenerationContext) -> GenerationView:
        async def on_cancel(interaction: discord.Interaction) -> None:
            await self._handle_cancel_request(context, interaction)

        async def on_reuse(interaction: discord.Interaction) -> None:
            await self._handle_reuse_request(context, interaction)

        return GenerationView(context.user.id, on_cancel, on_reuse)

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
        embed = self._build_generation_embed(context, status=status, title=title, color=color, extra_fields=extra_fields)
        kwargs: Dict[str, Any] = {"embed": embed}
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
        if context.model_preset_name:
            fields.append(("🧠 Model", context.model_preset_name, True))
        if context.lora_preset_name:
            fields.append(("🧩 LoRA", context.lora_preset_name, True))
        if context.resolution:
            fields.append(("🖼️ Resolution", context.resolution, True))
        if context.seed is not None:
            fields.append(("🌱 Seed", str(context.seed), True))
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
        if context.daily_limit is None:
            return None
        used = self.generation_store.get_count(context.user_id)
        return ui_embeds.format_usage_bar(used, context.daily_limit, reset_hint="resets in 24h")

    async def _handle_cancel_request(self, context: GenerationContext, interaction: discord.Interaction) -> None:
        if not interaction.response.is_done():
            await interaction.response.defer(ephemeral=True)
        if context.cancel_event.is_set():
            await interaction.followup.send("Generation already cancelled.", ephemeral=True)
            return
        logger.info("gen[%s|%s] cancellation requested", context.user_id, self._format_user_for_log(context.user))
        context.cancel_event.set()
        await self.queue.cancel_pending(context)
        if self.comfy_client and context.prompt_id:
            try:
                await self.comfy_client.cancel_prompt(context.prompt_id)
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Cancel prompt failed: %s", exc)
        await self._handle_cancelled_generation(context)
        self._finalize_generation_context(context, success=False)
        await interaction.followup.send("Generation cancelled.", ephemeral=True)

    async def _handle_reuse_request(self, context: GenerationContext, interaction: discord.Interaction) -> None:
        if interaction.user.id != int(context.user_id):
            await interaction.response.send_message("Only the original requester can reuse this prompt.", ephemeral=True)
            return
        last_request = self._last_requests.get(context.user_id)
        if not last_request:
            await interaction.response.send_message(
                "No previous request found to reuse. Run a generation first.", ephemeral=True
            )
            return
        if last_request.get("requires_image"):
            await interaction.response.send_message(
                "The last request used an image. Please run the command again with a new image to reuse it.",
                ephemeral=True,
            )
            return
        await self.handle_generation(
            interaction,
            last_request.get("workflow_type", "txt2img"),
            last_request.get("prompt") or "",
            workflow=last_request.get("workflow"),
            settings=last_request.get("settings"),
            resolution=last_request.get("resolution"),
            prompt_preset=last_request.get("prompt_preset"),
            model_preset=last_request.get("model_preset"),
            lora_preset=last_request.get("lora_preset"),
            seed=last_request.get("seed"),
        )

    async def _handle_cancelled_generation(self, context: GenerationContext, *, reason: str = "Generation cancelled by user.") -> None:
        if context.cancelled_notified:
            return
        context.cancelled_notified = True
        if context.view:
            context.view.disable()
        await self._update_generation_message(
            context, status=f"🛑 {reason}", title="🛑 Generation cancelled", color=ui_embeds.WARNING_COLOR
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
            self.generation_store.record_success(context.user_id, counted_usage=context.counted_usage)
        if context.counted_usage and not success and not context.cancel_event.is_set():
            new_value = max(0, self.generation_store.get_count(context.user_id) - 1)
            self.generation_store.set_count(context.user_id, new_value)
            try:
                channel = self._sync_channel_cache
                if channel:
                    asyncio.create_task(
                        self.sync_bridge.publish_limit(
                            channel=channel,
                            user_id=int(context.user_id),
                            used=new_value,
                            limit=context.daily_limit or self._public_tier().daily_limit or 0,
                            reset_at=float(self.generation_store.last_reset + 86400.0),
                            tier=context.tier,
                        )
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("SYNC publish rollback skipped: %s", exc)
        if context.slot_counted:
            try:
                channel = self._sync_channel_cache
                if channel:
                    asyncio.create_task(
                        self.sync_bridge.publish_active_delta(
                            channel=channel,
                            user_id=int(context.user_id),
                            delta=-1,
                            tier=context.tier,
                        )
                    )
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("SYNC active rollback skipped: %s", exc)

    # ---------------------- Messaging helpers ----------------------
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
        used = self.generation_store.get_count(str(interaction.user.id))
        usage = ui_embeds.format_usage_bar(used, limit, reset_hint="resets in 24h")
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

    async def _send_access_guild_required_message(self, interaction: discord.Interaction) -> None:
        embed = ui_embeds.build_notice_embed(
            title="🚫 Access restricted",
            description="Join the required server to start generations. https://discord.gg/XnxmanFBUp",
            color=ui_embeds.ERROR_COLOR,
        )
        await interaction.response.send_message(embed=embed, ephemeral=True)

    # ---------------------- Utilities ----------------------
    def _format_user_for_log(self, user: discord.abc.User) -> str:
        base_name = (
            getattr(user, "global_name", None)
            or getattr(user, "display_name", None)
            or getattr(user, "name", None)
            or str(user)
        )
        return f"{base_name} ({getattr(user, 'id', '?')})"
