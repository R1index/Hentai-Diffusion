import discord


class GenerationState:
    """Manages state for a single image generation"""

    def __init__(self, interaction: discord.Interaction, workflow_name: str, prompt: str, settings: str):
        self.interaction = interaction
        self.workflow_name = workflow_name
        self.prompt = prompt
        self.settings = settings
        self.message = None
        self.current_status = "Starting generation..."
        self.image_file = None

    def get_embed(self) -> discord.Embed:
        """Create embed for the current state"""
        embed = discord.Embed(title="🔨 RindexBot", color=0x2F3136)
        embed.add_field(name="Workflow", value=self.workflow_name, inline=True)
        if self.prompt:
            embed.add_field(name="Prompt", value=self.prompt, inline=False)
        if self.settings:
            embed.add_field(name="Settings", value=f"```{self.settings}```", inline=False)
        embed.add_field(name="Status", value=self.current_status, inline=False)
        return embed

    async def send_message(self) -> None:
        """Send a message without quoting"""
        embed = self.get_embed()
        try:
            self.message = await self.interaction.channel.send(embed=embed)  # Отправляет сообщение в канал
        except discord.Forbidden:
            await self.interaction.response.send_message(
                "У меня нет прав отправлять сообщения в этом канале.",
                ephemeral=True
            )
