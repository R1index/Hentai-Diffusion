"""Helpers for building Discord embeds used by the bot."""

from __future__ import annotations
from typing import Iterable, Optional, Sequence, Tuple

import discord

# Palette aligned with Discord brand colors.
ACCENT_COLOR = 0x5865F2
PROGRESS_COLOR = 0x3498DB
SUCCESS_COLOR = 0x57F287
WARNING_COLOR = 0xFEE75C
ERROR_COLOR = 0xED4245

EmbedField = Tuple[str, str, bool]


def _truncate_block(value: Optional[str], limit: int) -> Optional[str]:
    if not value:
        return None
    truncated = value[:limit]
    suffix = "…" if len(value) > limit else ""
    return f"```{truncated}{suffix}```"


def build_generation_embed(
    *,
    title: str,
    user: discord.abc.User,
    workflow_name: str,
    status: str,
    color: int = ACCENT_COLOR,
    prompt: Optional[str] = None,
    settings: Optional[str] = None,
    usage: Optional[str] = None,
    fields: Optional[Iterable[EmbedField]] = None,
    footer: str = "Support us ❤️ boosty.to/rindex",
) -> discord.Embed:
    """Create a consistent embed for generation-related updates."""

    author_name = getattr(user, "global_name", None) or getattr(user, "display_name", None) or getattr(user, "name", None) or "User"
    embed = discord.Embed(
        title=title,
        description=(
            f"**Workflow:** `{workflow_name}`\n"
            f"**Requested by:** {user.mention}"
        ),
        color=color,
        timestamp=discord.utils.utcnow(),
    )
    avatar = getattr(user, "display_avatar", None)
    if avatar:
        embed.set_author(name=author_name, icon_url=avatar.url)
        embed.set_thumbnail(url=avatar.url)

    embed.add_field(name="📊 Status", value=status, inline=False)

    if fields:
        for name, value, inline in fields:
            embed.add_field(name=name, value=value, inline=inline)

    if prompt:
        embed.add_field(name="🧠 Prompt", value=_truncate_block(prompt, 1000), inline=False)

    if settings:
        embed.add_field(name="⚙️ Settings", value=_truncate_block(settings, 500), inline=False)

    if usage:
        embed.add_field(name="📈 Usage", value=usage, inline=False)

    embed.set_footer(text=footer)
    return embed


def build_error_embed(*, user: discord.abc.User, workflow_name: str, error: str) -> discord.Embed:
    """Create an embed for unexpected failures."""

    status = f"❌ Error encountered:\n```{error[:1000]}```"
    return build_generation_embed(
        title="Generation failed",
        user=user,
        workflow_name=workflow_name,
        status=status,
        color=ERROR_COLOR,
    )


def build_notice_embed(*, title: str, description: str, color: int = ACCENT_COLOR) -> discord.Embed:
    """Build a simple embed with the shared footer."""

    embed = discord.Embed(
        title=title,
        description=description,
        color=color,
        timestamp=discord.utils.utcnow(),
    )
    embed.set_footer(text="Support us ❤️ boosty.to/rindex")
    return embed


def build_limit_embed(description: str) -> discord.Embed:
    """Build an embed used for limit and permission messages."""

    return build_notice_embed(
        title="⚠️ Access limited",
        description=description,
        color=WARNING_COLOR,
    )


def format_usage_bar(used: int, total: int, *, reset_hint: Optional[str] = None) -> str:
    """Format usage information with a text progress bar."""

    total = max(total, 1)
    used = max(0, min(used, total))
    filled = round((used / total) * 12)
    bar = "█" * filled + "░" * (12 - filled)
    percentage = int((used / total) * 100)
    parts: Sequence[str] = [f"**{used}/{total}** ({percentage}%)"]
    if reset_hint:
        parts.append(f"⏳ {reset_hint}")
    progress_line = f"`{bar}`"
    return " • ".join(parts) + "\n" + progress_line
