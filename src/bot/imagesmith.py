import os
import time
import yaml
import discord
import importlib
import inspect
import sys
import asyncio
import uuid
import hashlib
import hmac
import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict
from discord.ext import commands, tasks
from discord import app_commands

from logger import logger
from .commands import rgen_command, workflows_command
from ..comfy.workflow_manager import WorkflowManager
from ..core.hook_manager import HookManager
from ..core.generation_queue import GenerationQueue
from ..comfy.client import ComfyUIClient
from ..core.security import BasicSecurity, SecurityManager
from ..core.plugin import Plugin



class ComfyUIBot(commands.Bot):
    GENERATION_COUNTS_FILE = 'generation_counts.yml'
    
    def __init__(self, configuration_path: str = 'configuration.yml', plugins_path: str = 'plugins'):
        """Initialize the bot with configuration and plugins path"""
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.guilds = True

        super().__init__(command_prefix='/', intents=intents)

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
        self.active_generations: Dict[str, bool] = {}
        self.generation_start_times: Dict[str, float] = {}
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
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Load security lists and generation counts
        self._load_security_lists()
        self._load_generation_counts()
        
        # Start monitoring task
        self.monitor_generations.start()

    def _get_counts_path(self) -> str:
        """Get full path to generation counts file"""
        return os.path.join('data', self.GENERATION_COUNTS_FILE)

    def _load_security_lists(self) -> None:
        """Load blocked users, donors and supporter role configuration."""

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
                "Loaded security configuration: %d blocked users, %d donors",
                len(self.blocked_users),
                len(self.donor_users),
            )

            if self.access_guild_id:
                role_descriptor = self.supporter_role_id or self.supporter_role_name
                logger.info(
                    "Supporter role checks enabled for guild %s (role %s)",
                    self.access_guild_id,
                    role_descriptor,
                )
            else:
                logger.warning("Supporter role checks disabled: access_guild_id is not configured")

        except Exception as exc:
            logger.error(f"Failed to load security configuration: {exc}")
            self.blocked_users = set()
            self.donor_users = set()
            self.access_guild_id = None
            self.supporter_role_name = "Supporter"
            self.supporter_role_id = None

    def _load_generation_counts(self) -> None:
        """Load generation counts from file"""
        counts_path = self._get_counts_path()
        try:
            if os.path.exists(counts_path):
                with open(counts_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                    self.user_generation_counts = defaultdict(int, data.get('counts', {}))
                    self.last_reset_time = data.get('last_reset', time.time())
                    logger.info("Loaded generation counts from file")
            else:
                logger.info("No generation counts file found, creating new one")
                self._save_generation_counts()
        except Exception as e:
            logger.error(f"Failed to load generation counts: {e}")
            self.user_generation_counts = defaultdict(int)
            self.last_reset_time = time.time()
            self._save_generation_counts()

    def _save_generation_counts(self) -> None:
        """Save generation counts to file"""
        counts_path = self._get_counts_path()
        try:
            data = {
                'counts': dict(self.user_generation_counts),
                'last_reset': self.last_reset_time
            }
            with open(counts_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(data, f, allow_unicode=True)
            logger.debug("Saved generation counts to file")
        except Exception as e:
            logger.error(f"Failed to save generation counts: {e}")

    def _reset_counts_if_needed(self) -> None:
        """Reset generation counts every 24 hours if needed"""
        current_time = time.time()
        if current_time - self.last_reset_time >= 86400:  # 24 hours
            self.user_generation_counts.clear()
            self.last_reset_time = current_time
            self._save_generation_counts()
            logger.info("Reset user generation counts")

    @tasks.loop(minutes=5)
    async def monitor_generations(self):
        """Monitor active generations and clear stuck ones"""
        current_time = time.time()
        stuck_threshold = 1800  # 30 minutes timeout
        
        stuck_users = [
            user_id for user_id, start_time in self.generation_start_times.items()
            if current_time - start_time > stuck_threshold
        ]
        
        if stuck_users:
            logger.warning(f"Found {len(stuck_users)} stuck generations, clearing...")
            for user_id in stuck_users:
                self.active_generations.pop(user_id, None)
                self.generation_start_times.pop(user_id, None)
                logger.info(f"Cleared stuck generation for user {user_id}")

    @monitor_generations.before_loop
    async def before_monitor(self):
        await self.wait_until_ready()

    async def close(self):
        """Override close to save data and stop tasks"""
        self.monitor_generations.cancel()
        self._save_generation_counts()
        await super().close()

    def _format_time_remaining(self) -> str:
        """Format remaining time until daily reset"""
        current_time = time.time()
        remaining = 86400 - (current_time - self.last_reset_time)
        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)
        return f"{hours}h {minutes}m"


    async def setup_hook(self) -> None:
        """Setup the bot before starting"""
        logger.info("Setting up bot...")
        
        # Load plugins
        await self._load_plugins()
        
        # Initialize ComfyUI client
        try:
            await self.hook_manager.execute_hook(
                'is.comfyui.client.before_create', 
                self.workflow_manager.config['comfyui']['instances']
            )
            
            self.comfy_client = ComfyUIClient(
                self.workflow_manager.config['comfyui']['instances'], 
                self.hook_manager
            )
            
            await self.hook_manager.execute_hook(
                'is.comfyui.client.after_create', 
                self.workflow_manager.config['comfyui']['instances']
            )
            
            await self.comfy_client.connect()
            logger.info("Connected to ComfyUI")
        except Exception as e:
            logger.error(f"Failed to connect to ComfyUI: {e}")
            await self._cleanup_resources()
            sys.exit(1)

        # Register commands
        logger.info("Registering commands...")
        try:
            self.tree.add_command(rgen_command(self))
            self.tree.add_command(workflows_command(self))
            self.tree.add_command(self._create_limits_command())

            commands = await self.tree.sync()
            logger.info(f"Registered {len(commands)} commands:")
            for cmd in commands:
                logger.info(f"- /{cmd.name}")
        except Exception as e:
            logger.error(f"Failed to sync commands: {e}")
            await self._cleanup_resources()
            sys.exit(1)

    async def on_message(self, message: discord.Message):
        if message.author.bot is False:
            return
        if message.channel.id != self.SYNC_CHANNEL_ID:
            return
        if not message.content.startswith(self.SYNC_PREFIX):
            return

        try:
            data = json.loads(message.content[len(self.SYNC_PREFIX):])
            if not self._sync_verify(data):
                logger.warning("SYNC: signature mismatch — ignored")
                return

            payload = data["payload"]
            event_id = payload["event_id"]
            ts = int(payload["ts"])
            src_bot = int(payload["source_bot_id"])

            self._sync_prune_seen()
            if event_id in self._sync_seen:
                return
            self._sync_seen[event_id] = self._sync_now() + self._sync_seen_ttl

            if src_bot == self.user.id:
                return

            user_id = str(int(payload["user_id"]))
            used = int(payload["generations_used"])
            limit = int(payload["limit"])
            reset_at = float(payload["reset_at"])
            incoming_last_reset = reset_at - 86400.0

            if getattr(self, "last_reset_time", 0) > incoming_last_reset + 2:
                logger.debug("SYNC: local reset is newer — skip")
                return

            self.user_generation_counts[user_id] = used
            self.last_reset_time = incoming_last_reset

            if hasattr(self, "_counts_dirty"):
                self._counts_dirty = True
            else:
                self._save_generation_counts()

            logger.info(f"SYNC: applied from bot {src_bot} → user {user_id}: {used}/{limit}, reset_at={int(reset_at)}")

        except Exception as e:
            logger.error(f"SYNC parse/apply failed: {e}", exc_info=True)

    async def on_ready(self) -> None:
        """Called when the bot is ready"""
        logger.info(f'Logged in as {self.user} (ID: {self.user.id})')
        logger.info(f'Connected to {len(self.guilds)} guilds')

        # Generate invite link
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
            scopes=("bot", "applications.commands")
        )

        logger.info(f"Invite link: {invite_link}")
        logger.info("Bot is ready!")

    async def _cleanup_resources(self) -> None:
        """Clean up resources before shutdown"""
        logger.info("Cleaning up resources...")
        try:
            if self.comfy_client:
                await self.comfy_client.close()
            await self.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    async def _load_plugins(self) -> None:
        """Load all plugins from plugins directory"""
        plugins_dir = Path(self.plugins_path)
        if not plugins_dir.exists():
            logger.warning("No plugins directory found")
            return

        sys.path.append(str(Path.cwd()))
        plugin_files = [f for f in plugins_dir.glob("*.py") if f.name != "__init__.py"]

        for plugin_file in plugin_files:
            logger.info(f"Loading plugin: {plugin_file}")
            try:
                spec = importlib.util.spec_from_file_location(plugin_file.stem, plugin_file)
                if spec is None:
                    logger.warning(f"Failed to get spec for {plugin_file}")
                    continue

                module = importlib.util.module_from_spec(spec)
                if spec.loader is None:
                    logger.warning(f"Failed to get loader for {plugin_file}")
                    continue

                spec.loader.exec_module(module)
                logger.debug(f"Successfully loaded module: {module.__name__}")

                for item_name in dir(module):
                    if item_name.startswith('__'):
                        continue

                    try:
                        obj = getattr(module, item_name)
                        if inspect.isclass(obj):
                            if issubclass(obj, Plugin) and obj != Plugin:
                                plugin_instance = obj(self)
                                await plugin_instance.on_load()
                                self.plugins.append(plugin_instance)
                                logger.info(f"Loaded plugin: {obj.__name__}")
                    except Exception as e:
                        logger.error(f"Error processing item {item_name}: {e}")

            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_file}: {e}")

        logger.info(f"Loaded {len(self.plugins)} plugins")
        
    async def _has_unlimited_access(self, interaction: discord.Interaction) -> bool:
        """
        True — если у пользователя есть Supporter-роль на сервере access_guild_id.
        Работает и в DM: ищем участника на целевом сервере. Пишем подробные логи.
        """
        try:
            if not self.access_guild_id:
                logger.info("Supporter check: access_guild_id is empty")
                return False

            try:
                gid = int(self.access_guild_id)
            except Exception:
                logger.warning(f"Supporter check: invalid access_guild_id={self.access_guild_id!r}")
                return False

            # 1) Ищем гильдию
            if interaction.guild and interaction.guild.id == gid:
                target_guild = interaction.guild
                logger.debug(f"Supporter check: using interaction.guild={gid}")
            else:
                target_guild = self.get_guild(gid)
                if target_guild is None:
                    try:
                        target_guild = await self.fetch_guild(gid)
                        logger.debug(f"Supporter check: fetched guild {gid}")
                    except Exception as e:
                        logger.warning(f"Supporter check: fetch_guild({gid}) failed: {e}")
                        return False

            # 2) Ищем участника
            member = target_guild.get_member(interaction.user.id)
            if member is None:
                try:
                    member = await target_guild.fetch_member(interaction.user.id)
                    logger.debug(f"Supporter check: fetched member {interaction.user.id} on guild {gid}")
                except Exception as e:
                    logger.info(f"Supporter check: user {interaction.user.id} is not in guild {gid}: {e}")
                    return False

            # 3) Ищем роль: по ID (приоритет), затем по имени (без регистра)
            role = None
            rid = None

            if getattr(self, "supporter_role_id", None):
                try:
                    rid = int(self.supporter_role_id)
                    role = target_guild.get_role(rid)
                    if role:
                        logger.debug(f"Supporter check: found role by ID {rid}: {role.name}")
                except Exception:
                    role = None

            if role is None and getattr(self, "supporter_role_name", None):
                lname = self.supporter_role_name.casefold()
                for r in target_guild.roles:
                    if r.name.casefold() == lname:
                        role = r
                        rid = r.id
                        logger.debug(f"Supporter check: found role by name '{self.supporter_role_name}' -> {r.id}")
                        break

            if not role:
                logger.warning(
                    "Supporter check: role not found on guild "
                    f"(guild_id={gid}, role_id={getattr(self,'supporter_role_id',None)}, "
                    f"role_name={getattr(self,'supporter_role_name',None)!r})"
                )
                return False

            member_role_ids = {r.id for r in getattr(member, "roles", [])}
            has_role = role.id in member_role_ids
            logger.info(
                f"Supporter check: guild={gid}, member={member.id}, role={role.id}/{role.name}, "
                f"member_roles={sorted(member_role_ids)}, has={has_role}"
            )
            return has_role

        except Exception as e:
            logger.error(f"Supporter role check failed: {e}", exc_info=True)
            return False        

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
        for k, exp in list(self._sync_seen.items()):
            if exp <= now:
                self._sync_seen.pop(k, None)

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
                "source_bot_id": self.user.id,
                "user_id": int(user_id),
                "generations_used": int(used),
                "limit": int(limit),
                "reset_at": float(reset_at),
            }
            wrapper = {"payload": payload, "sig": self._sync_sign(payload)}
            content = self.SYNC_PREFIX + self._sync_canon(wrapper)
            await channel.send(content)
            self._sync_seen[event_id] = self._sync_now() + self._sync_seen_ttl
            self._sync_prune_seen()
        except Exception as e:
            logger.error(f"publish_limit_update failed: {e}", exc_info=True)

    def _create_limits_command(self):
        """Create the limits slash command"""
        @app_commands.command(
            name="limits",
            description="Check your current generation limits and status"
        )
        async def limits_command(interaction: discord.Interaction):
            """Check generation limits command handler"""
            user_id = str(interaction.user.id)
            self._reset_counts_if_needed()
            
            is_donor = user_id in self.donor_users
            count = self.user_generation_counts.get(user_id, 0)
            
            embed = discord.Embed(
                title="🔢 Your Generation Limits",
                color=0x7289DA
            )
            
            if is_donor:
                embed.description = "🌟 **Donor Status**: Unlimited access to all workflows!"
                embed.add_field(
                    name="💎 Donor Benefits",
                    value="✅ Unlimited generations\n✅ Priority access\n✅ All workflows",
                    inline=False
                )
            else:
                embed.description = (
                    f"🔄 **Generations used**: {count}/50 (resets in {self._format_time_remaining()})\n"
                    "💎 Become a donor for unlimited access!"
                )
                embed.add_field(
                    name="ℹ️ Information",
                    value="The 'forge' workflow is always unlimited for all users",
                    inline=False
                )
            
            embed.set_footer(text="Support us ❤️ boosty.to/rindex")
            
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
        """Handle image generation requests while enforcing per-user limits."""

        user_id = str(interaction.user.id)

        try:
            self._reset_counts_if_needed()

            if user_id in self.blocked_users:
                await self._send_blocked_message(interaction)
                return

            if self.active_generations.get(user_id):
                await self._send_active_generation_message(interaction)
                return

            logger.info(
                "Handling generation request: workflow_type=%s, workflow=%s", workflow_type, workflow
            )

            is_supporter = await self._has_unlimited_access(interaction)
            is_donor = is_supporter or user_id in self.donor_users

            if not is_donor:
                logger.info("User %s is subject to limits", user_id)
                if self.user_generation_counts[user_id] >= 50:
                    await self._send_limit_reached_message(interaction)
                    return

                self.user_generation_counts[user_id] += 1
                self._save_generation_counts()

                try:
                    asyncio.create_task(
                        self._publish_limit_update(
                            user_id=int(user_id),
                            used=int(self.user_generation_counts[user_id]),
                            limit=50,
                            reset_at=float(self.last_reset_time + 86400.0),
                        )
                    )
                except Exception as exc:
                    logger.debug(f"SYNC publish skipped: {exc}")

            self.active_generations[user_id] = True
            self.generation_start_times[user_id] = time.time()

            logger.info(
                "Generation started by %s (Unlimited: %s, Count: %s, Workflow: %s)",
                interaction.user,
                is_donor,
                self.user_generation_counts[user_id],
                workflow or "default",
            )

            await self._process_generation(
                interaction, workflow_type, prompt, workflow, settings, input_image, is_donor
            )

        except Exception as exc:
            self.active_generations.pop(user_id, None)
            self.generation_start_times.pop(user_id, None)
            if not interaction.response.is_done():
                await self._send_error_message(interaction, str(exc))
            logger.error(f"Generation error: {exc}", exc_info=True)
            raise

    async def _send_blocked_message(self, interaction: discord.Interaction) -> None:
        """Send message to blocked users"""
        await interaction.response.send_message(
            embed=discord.Embed(
                title="🚫 Access Restricted",
                description="Your account has been blocked from using this bot.\n\n"
                           "If you believe this is a mistake, please contact support.",
                color=0xFF0000
            ).set_footer(text="Support: boosty.to/rindex"),
            ephemeral=True
        )

    async def _send_active_generation_message(self, interaction: discord.Interaction) -> None:
        """Send message when user has active generation"""
        await interaction.response.send_message(
            embed=discord.Embed(
                title="⏳ Already Processing",
                description="You have an active generation in progress.\n"
                           "Please wait for it to complete before starting a new one.",
                color=0xF5A623
            ).set_footer(text="Patience is a virtue ❤️"),
            ephemeral=True
        )

    async def _send_limit_reached_message(self, interaction: discord.Interaction) -> None:
        """Send message when generation limit is reached"""
        await interaction.response.send_message(
            embed=discord.Embed(
                title="⚠️ Generation Limit Reached",
                description="You've reached your 24-hour generation limit (50 generations).\n\n"
                          "Options:\n"
                          "- Become a donor for unlimited access",
                color=0xFFA500
            ).set_footer(text="Support us ❤️ boosty.to/rindex"),
            ephemeral=True
        )

    async def _send_error_message(self, interaction: discord.Interaction, error: str) -> None:
        """Send error message"""
        await interaction.response.send_message(
            embed=discord.Embed(
                title="❌ Unexpected Error",
                description=f"```{error[:1000]}```",
                color=0xFF0000
            ).set_footer(text="Please try again or contact support ❤️ boosty.to/rindex")
        )

    async def _process_generation(
        self,
        interaction: discord.Interaction,
        workflow_type: str,
        prompt: str,
        workflow: Optional[str],
        settings: Optional[str],
        input_image: Optional[discord.Attachment],
        is_donor: bool,
    ) -> None:
        """Process the actual generation request"""
        user_id = str(interaction.user.id)
        workflow_name = workflow or self.workflow_manager.get_default_workflow(workflow_type)
        workflow_config = self.workflow_manager.get_workflow(workflow_name)

        # Security checks
        security_results = await self.hook_manager.execute_hook(
            'is.security',
            interaction, workflow_name, workflow_type, prompt, workflow_config, settings
        )

        for result in security_results:
            if not result.state:
                await interaction.response.send_message(
                    embed=discord.Embed(
                        title="❌ Security Check Failed",
                        description=result.message,
                        color=0xFF0000
                    ).set_footer(text="Contact support ❤️ boosty.to/rindex")
                )
                self.active_generations.pop(user_id, None)
                self.generation_start_times.pop(user_id, None)
                return

        # Validate workflow
        if not workflow_config:
            await interaction.response.send_message(
                embed=discord.Embed(
                    title="❌ Workflow Not Found",
                    description=f"Workflow '{workflow_name}' not found!",
                    color=0xFF0000
                ).set_footer(text="Available workflows: /workflows")
            )
            self.active_generations.pop(user_id, None)
            self.generation_start_times.pop(user_id, None)
            return

        if workflow_config.get('type', 'txt2img') != workflow_type:
            await interaction.response.send_message(
                embed=discord.Embed(
                    title="❌ Workflow Type Mismatch",
                    description=f"Workflow '{workflow_name}' is not a {workflow_type} workflow!",
                    color=0xFF0000
                ).set_footer(text="Check workflow types with /workflows")
            )
            self.active_generations.pop(user_id, None)
            self.generation_start_times.pop(user_id, None)
            return

        # Handle image input
        image_data = None
        if workflow_type in ['img2img', 'upscale']:
            if not input_image or not input_image.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                await interaction.response.send_message(
                    embed=discord.Embed(
                        title="❌ Invalid Image",
                        description="Valid input image is required for this workflow type! (PNG, JPG, JPEG, WEBP)",
                        color=0xFF0000
                    ).set_footer(text="Supported formats: PNG, JPG, JPEG, WEBP")
                )
                self.active_generations.pop(user_id, None)
                self.generation_start_times.pop(user_id, None)
                return
            image_data = await input_image.read()

        # Create initial response
        queue_position = self.generation_queue.get_queue_position()
        embed = discord.Embed(
            title="🖼️ Generation Started", 
            color=0x2F3136,
            description=f"**Workflow:** `{workflow_name}`\n"
                      f"**User:** {interaction.user.mention}"
        )
        
        if is_donor:
            embed.description += "\n🌟 **Donor Status**: Unlimited"
        else:
            embed.description += f"\n🔄 **Usage**: {self.user_generation_counts[user_id]}/50"
        
        embed.set_author(name="RindexBot")
        embed.add_field(name="🔮 Prompt", value=f"```{prompt[:1000]}```" if prompt else "No prompt", inline=False)
        
        if settings:
            embed.add_field(name="⚙️ Settings", value=f"```{settings[:500]}```", inline=False)
            
        embed.add_field(
            name="📊 Status", 
            value=f"⏳ Queued (Position: {queue_position + 1})" if queue_position > 0 else "🚀 Starting generation...", 
            inline=False
        )
        
        embed.set_footer(text="Support us ❤️ boosty.to/rindex")
        await interaction.response.send_message(embed=embed)
        message = await interaction.original_response()

        # Run the generation
        async def run_generation():
            try:
                workflow_json = self.workflow_manager.prepare_workflow(
                    workflow_name, prompt, settings, image_data
                )

                result = await self.comfy_client.generate(workflow_json)
                if 'error' in result:
                    raise Exception(result['error'])

                prompt_id = result.get('prompt_id')
                if not prompt_id:
                    raise Exception("No prompt ID received from ComfyUI")

                async def update_message(status: str, image_file: Optional[discord.File] = None, limit: Optional[str] = None):
                    embed = discord.Embed(
                        title="🔄 Generation Progress",
                        color=0x7289DA,
                        description=f"**Workflow:** `{workflow_name}`\n"
                                    f"**User:** {interaction.user.mention}"
                    )

                    embed.add_field(name="📊 Status", value=status, inline=False)

                    if prompt:
                        embed.add_field(name="🔮 Prompt", value=f"```{prompt[:1000]}```", inline=False)

                    if settings:
                        embed.add_field(name="⚙️ Settings", value=f"```{settings[:500]}```", inline=False)

                    if limit:
                        try:
                            used, total = map(int, limit.split("/"))
                            percent = used / total
                            bar_length = 20
                            filled_length = int(bar_length * percent)
                            bar = "█" * filled_length + "—" * (bar_length - filled_length)
                            embed.add_field(
                                name="📈 Generation Limit",
                                value=f"`{used}/{total}` • {int(percent * 100)}%\n`[{bar}]`",
                                inline=False
                            )
                        except Exception as e:
                            embed.add_field(name="📈 Limit", value=f"```{limit}```", inline=False)

                    embed.set_footer(text="Support us ❤️ boosty.to/rindex")

                    if image_file:
                        await message.edit(embed=embed, attachments=[image_file])
                    else:
                        await message.edit(embed=embed)

                await self.comfy_client.listen_for_updates(prompt_id, update_message)

            except Exception as e:
                error_embed = discord.Embed(
                    title="❌ Generation Failed", 
                    color=0xFF0000,
                    description=f"**Workflow:** `{workflow_name}`\n"
                              f"**Error:** ```{str(e)[:1000]}```"
                )
                error_embed.set_author(name="RindexBot")
                error_embed.add_field(name="👤 User", value=interaction.user.mention, inline=True)
                error_embed.set_footer(text="Contact support ❤️ boosty.to/rindex")
                await message.edit(embed=error_embed)

            finally:
                self.active_generations.pop(user_id, None)
                self.generation_start_times.pop(user_id, None)

        await self.generation_queue.add_to_queue(run_generation)