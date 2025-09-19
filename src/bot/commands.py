from typing import Optional

import discord
from discord import app_commands


def rgen_command(bot):
    """Create the forge command for txt2img generation"""

    resolution_choices = [
        app_commands.Choice(name=label, value=value)
        for label, value in bot.workflow_manager.get_resolution_presets()[:25]
    ]

    @app_commands.command(
        name="rgen",
        description="Forge an image using text-to-image"
    )
    @app_commands.describe(
        prompt="Description of the image you want to create",
        resolution="Select the output resolution (optional)",
        workflow="The workflow to use (optional)",
        settings="Additional settings (optional)"
    )
    async def rgen(
            interaction: discord.Interaction,
            prompt: str,
            resolution: Optional[app_commands.Choice[str]] = None,
            workflow: Optional[str] = None,
            settings: Optional[str] = None
    ):
        selected_resolution = resolution.value if resolution else None
        await bot.handle_generation(
            interaction,
            'txt2img',
            prompt,
            workflow,
            settings,
            resolution=selected_resolution,
        )

    if resolution_choices:
        rgen = app_commands.choices(resolution=resolution_choices)(rgen)

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
        await bot.handle_generation(
            interaction,
            'img2img',
            prompt,
            workflow,
            settings,
            input_image=image,
        )

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
        await bot.handle_generation(
            interaction,
            'upscale',
            prompt,
            workflow,
            settings,
            input_image=image,
        )

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


def profile_command(bot):
    """Create the profile command for viewing personal statistics."""

    @app_commands.command(
        name="profile",
        description="Показать статистику генераций и статус спонсорства",
    )
    async def profile(interaction: discord.Interaction) -> None:
        user_id = str(interaction.user.id)
        stats = bot.get_user_generation_summary(user_id)

        supporter_role = await bot._has_unlimited_access(interaction)
        listed_donor = user_id in bot.donor_users

        status_details = []
        if listed_donor:
            status_details.append("в списке доноров")
        if supporter_role:
            status_details.append("есть роль поддержки")

        if status_details:
            sponsorship_status = f"💎 Активен ({', '.join(status_details)})"
        else:
            sponsorship_status = "🪙 Не активен"

        embed = discord.Embed(
            title="👤 Профиль пользователя",
            description=f"Статистика для {interaction.user.mention}",
            color=0x5865F2,
        )

        if interaction.user.display_avatar:
            embed.set_thumbnail(url=interaction.user.display_avatar.url)

        stats_lines = [
            f"Сегодня: **{stats['day']}**",
            f"7 дней: **{stats['week']}**",
            f"30 дней: **{stats['month']}**",
            f"Всего: **{stats['total']}**",
        ]
        embed.add_field(name="📈 Генерации", value="\n".join(stats_lines), inline=False)
        embed.add_field(name="💖 Спонсорство", value=sponsorship_status, inline=False)
        embed.set_footer(text="Support us ❤️ boosty.to/rindex")

        await interaction.response.send_message(embed=embed, ephemeral=True)

    return profile
