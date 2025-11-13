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

    def __init__(self, owner_id: int, cancel_callback: Callable[[discord.Interaction], Awaitable[None]]):
        super().__init__(timeout=None)
        self.owner_id = owner_id
        self._cancel_callback = cancel_callback

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.danger, emoji="🛑")
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):  # type: ignore[override]
        if interaction.user.id != self.owner_id:
            await interaction.response.send_message(
                "You can only cancel your own generation.",
                ephemeral=True,
            )
            return

        await self._cancel_callback(interaction)

    def disable(self) -> None:
        """Disable all controls in the view."""

        for item in self.children:
            item.disabled = True
