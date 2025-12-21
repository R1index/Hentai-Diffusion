from __future__ import annotations

from typing import Awaitable, Callable

import discord

from .buttons import ImageButton


class ImageView(discord.ui.View):
    """Custom view for image interaction buttons."""

    def __init__(self, prompt_id: str, has_upscaler: bool = False):
        super().__init__(timeout=None)

        if has_upscaler:
            self.add_item(ImageButton("Upscale", f"upscale_{prompt_id}", "✨"))

        self.add_item(ImageButton("Regenerate", f"regenerate_{prompt_id}", "🔄"))
        self.add_item(ImageButton("Use as Input", f"img2img_{prompt_id}", "🖼"))


class GenerationView(discord.ui.View):
    """View with controls for an active generation."""

    def __init__(
            self,
            owner_id: int,
            cancel_callback: Callable[[discord.Interaction], Awaitable[None]],
            reuse_callback: Callable[[discord.Interaction], Awaitable[None]] | None = None,
    ):
        super().__init__(timeout=None)
        self.owner_id = owner_id
        self._cancel_callback = cancel_callback
        self._reuse_callback = reuse_callback

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.danger, emoji="🛑")
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):  # type: ignore[override]
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(
                "You can only cancel your own generation.",
                ephemeral=True,
            )
            return

        await self._cancel_callback(interaction)

    @discord.ui.button(label="Reuse request", style=discord.ButtonStyle.secondary, emoji="🔁")
    async def reuse(self, interaction: discord.Interaction, button: discord.ui.Button):  # type: ignore[override]
        if not self._reuse_callback:
            await interaction.response.send_message(
                "This request cannot be reused.",
                ephemeral=True,
            )
            return

        await self._reuse_callback(interaction)

    def disable(self) -> None:
        """Disable all controls in the view."""

        for item in self.children:
            if isinstance(item, discord.ui.Button) and item.label == "Reuse request":
                # Keep the reuse control available after completion.
                continue
            item.disabled = True
