from typing import Optional

import discord
from discord import app_commands


def rgen_command(bot):
    """Create the forge command for txt2img generation"""

    @app_commands.command(
        name="rgen",
        description="Forge an image using text-to-image"
    )
    @app_commands.describe(
        prompt="Description of the image you want to create",
        workflow="The workflow to use (optional)",
        settings="Additional settings (optional)"
    )
    async def rgen(
            interaction: discord.Interaction,
            prompt: str,
            workflow: Optional[str] = None,
            settings: Optional[str] = None
    ):
        await bot.handle_generation(interaction, 'txt2img', prompt, workflow, settings)

    return rgen


def reforge_command(bot):
    """Create the reforge command for img2img generation"""

    @app_commands.command(
        name="reforge",
        description="Reforge an existing image using image-to-image"
    )
    @app_commands.describe(
        image="The image to reforge",
        prompt="Description of the changes you want to make",
        workflow="The workflow to use (optional)",
        settings="Additional settings (optional)"
    )
    async def reforge(
            interaction: discord.Interaction,
            image: discord.Attachment,
            prompt: str,
            workflow: Optional[str] = None,
            settings: Optional[str] = None
    ):
        await bot.handle_generation(interaction, 'img2img', prompt, workflow, settings, image)

    return reforge


def upscale_command(bot):
    """Create the upscale command"""

    @app_commands.command(
        name="upscale",
        description="Upscale an existing image"
    )
    @app_commands.describe(
        image="The image to upscale",
        prompt="Description of the changes you want to make",
        workflow="The workflow to use (optional)",
        settings="Additional settings (optional)"
    )
    async def upscale(
            interaction: discord.Interaction,
            image: discord.Attachment,
            prompt: str,
            workflow: Optional[str] = None,
            settings: Optional[str] = None
    ):
        await bot.handle_generation(interaction, 'upscale', prompt, workflow, settings, image)

    return upscale


def workflows_command(bot):
    """Create the workflows command"""

    @app_commands.command(
        name="workflows",
        description="List available workflows"
    )
    @app_commands.describe(
        type="Type of workflows to list (txt2img, img2img, upscale)"
    )
    async def workflows(
            interaction: discord.Interaction,
            type: Optional[str] = None
    ):
        workflows = bot.workflow_manager.get_selectable_workflows(type)

        # Filter out workflows with "landscape" in their names
        filtered_workflows = {name: workflow for name, workflow in workflows.items() if "landscape" not in name.lower()}

        embeds = []
        embed = discord.Embed(
            title="📋 Available Workflows",
            color=0x2F3136
        )

        if type:
            embed.description = f"Showing {type} workflows"

        field_count = 0
        for name, workflow in filtered_workflows.items():
            workflow_type = workflow.get('type', 'txt2img')
            description = workflow.get('description', 'No description')
            if field_count >= 25:
                embeds.append(embed)
                embed = discord.Embed(
                    title="📋 Available Workflows (continued)",
                    color=0x2F3136
                )
                if type:
                    embed.description = f"Showing {type} workflows (continued)"
                field_count = 0

            embed.add_field(
                name=f"__{name}__  or  __{name}landscape__",
                value=description,
                inline=False
            )
            field_count += 1

        if embed.fields:
            embeds.append(embed)

        # Отправляем первый ответ
        await interaction.response.send_message(embed=embeds[0])

        # Отправляем оставшиеся вложения
        for embed in embeds[1:]:
            await interaction.followup.send(embed=embed)

    return workflows
