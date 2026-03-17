from typing import Callable, Optional

import discord
from discord import app_commands


def _make_preset_autocomplete(
        search_func: Callable[[str, int], list]
) -> Callable[[discord.Interaction, str], list[app_commands.Choice[str]]]:
    async def _autocomplete(
            interaction: discord.Interaction, current: str
    ) -> list[app_commands.Choice[str]]:
        presets = search_func(current, limit=25)
        return [
            app_commands.Choice(name=preset.name, value=preset.value)
            for preset in presets
        ]

    return _autocomplete


def _preset_autocompletes(bot):
    return (
        _make_preset_autocomplete(bot.workflow_manager.search_prompt_presets),
        _make_preset_autocomplete(bot.workflow_manager.search_model_presets),
        _make_preset_autocomplete(bot.workflow_manager.search_lora_presets),
    )


def rgen_command(bot):
    """Create the forge command for txt2img generation"""

    resolution_choices = [
        app_commands.Choice(name=label, value=value)
        for label, value in bot.workflow_manager.get_resolution_presets()[:25]
    ]

    prompt_autocomplete, model_autocomplete, lora_autocomplete = _preset_autocompletes(bot)

    @app_commands.command(
        name="rgen",
        description="Forge an image using text-to-image"
    )
    @app_commands.describe(
        prompt="Description of the image you want to create",
        prompt_preset="Select a prompt preset (optional)",
        model_preset="Select a model preset (optional)",
        lora_preset="Select a LoRA preset (optional)",
        seed="Seed value (optional)",
        resolution="Select the output resolution (optional)",
        workflow="The workflow to use (optional)",
        settings="Additional settings (optional)"
    )
    async def rgen(
            interaction: discord.Interaction,
            prompt: str,
            prompt_preset: Optional[str] = None,
            model_preset: Optional[str] = None,
            lora_preset: Optional[str] = None,
            seed: Optional[int] = None,
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
            prompt_preset=prompt_preset,
            model_preset=model_preset,
            lora_preset=lora_preset,
            seed=seed,
        )

    if resolution_choices:
        rgen = app_commands.choices(resolution=resolution_choices)(rgen)
    rgen = app_commands.autocomplete(prompt_preset=prompt_autocomplete)(rgen)
    rgen = app_commands.autocomplete(model_preset=model_autocomplete)(rgen)
    rgen = app_commands.autocomplete(lora_preset=lora_autocomplete)(rgen)

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
        prompt_preset="Select a prompt preset (optional)",
        model_preset="Select a model preset (optional)",
        lora_preset="Select a LoRA preset (optional)",
        seed="Seed value (optional)",
        workflow="The workflow to use (optional)",
        settings="Additional settings (optional)"
    )
    async def reforge(
            interaction: discord.Interaction,
            image: discord.Attachment,
            prompt: str,
            prompt_preset: Optional[str] = None,
            model_preset: Optional[str] = None,
            lora_preset: Optional[str] = None,
            seed: Optional[int] = None,
            workflow: Optional[str] = None,
        settings: Optional[str] = None
    ):
        await bot.handle_generation(
            interaction,
            'img2img',
            prompt,
            workflow,
            settings,
            prompt_preset=prompt_preset,
            model_preset=model_preset,
            lora_preset=lora_preset,
            seed=seed,
            input_image=image,
        )

    prompt_autocomplete, model_autocomplete, lora_autocomplete = _preset_autocompletes(bot)
    reforge = app_commands.autocomplete(prompt_preset=prompt_autocomplete)(reforge)
    reforge = app_commands.autocomplete(model_preset=model_autocomplete)(reforge)
    return app_commands.autocomplete(lora_preset=lora_autocomplete)(reforge)


def img2img_command(bot):
    """Create the img2img command for image-to-image generation"""

    @app_commands.command(
        name="img2img",
        description="Generate an image from an attached source image"
    )
    @app_commands.describe(
        image="Source image",
        prompt="Description of the changes you want to make",
        prompt_preset="Select a prompt preset (optional)",
        model_preset="Select a model preset (optional)",
        lora_preset="Select a LoRA preset (optional)",
        seed="Seed value (optional)",
        workflow="The workflow to use (optional)",
        settings="Additional settings (optional)"
    )
    async def img2img(
            interaction: discord.Interaction,
            image: discord.Attachment,
            prompt: str,
            prompt_preset: Optional[str] = None,
            model_preset: Optional[str] = None,
            lora_preset: Optional[str] = None,
            seed: Optional[int] = None,
            workflow: Optional[str] = None,
            settings: Optional[str] = None
    ):
        await bot.handle_generation(
            interaction,
            'img2img',
            prompt,
            workflow,
            settings,
            prompt_preset=prompt_preset,
            model_preset=model_preset,
            lora_preset=lora_preset,
            seed=seed,
            input_image=image,
        )

    prompt_autocomplete, model_autocomplete, lora_autocomplete = _preset_autocompletes(bot)
    img2img = app_commands.autocomplete(prompt_preset=prompt_autocomplete)(img2img)
    img2img = app_commands.autocomplete(model_preset=model_autocomplete)(img2img)
    return app_commands.autocomplete(lora_preset=lora_autocomplete)(img2img)


def upscale_command(bot):
    """Create the upscale command"""

    @app_commands.command(
        name="upscale",
        description="Upscale an existing image"
    )
    @app_commands.describe(
        image="The image to upscale",
        prompt="Description of the changes you want to make",
        prompt_preset="Select a prompt preset (optional)",
        model_preset="Select a model preset (optional)",
        lora_preset="Select a LoRA preset (optional)",
        seed="Seed value (optional)",
        workflow="The workflow to use (optional)",
        settings="Additional settings (optional)"
    )
    async def upscale(
            interaction: discord.Interaction,
            image: discord.Attachment,
            prompt: str,
            prompt_preset: Optional[str] = None,
            model_preset: Optional[str] = None,
            lora_preset: Optional[str] = None,
            seed: Optional[int] = None,
            workflow: Optional[str] = None,
        settings: Optional[str] = None
    ):
        await bot.handle_generation(
            interaction,
            'upscale',
            prompt,
            workflow,
            settings,
            prompt_preset=prompt_preset,
            model_preset=model_preset,
            lora_preset=lora_preset,
            seed=seed,
            input_image=image,
        )

    prompt_autocomplete, model_autocomplete, lora_autocomplete = _preset_autocompletes(bot)
    upscale = app_commands.autocomplete(prompt_preset=prompt_autocomplete)(upscale)
    upscale = app_commands.autocomplete(model_preset=model_autocomplete)(upscale)
    return app_commands.autocomplete(lora_preset=lora_autocomplete)(upscale)


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

        # Send the first response
        await interaction.response.send_message(embed=embeds[0])

        # Send the remaining embeds
        for embed in embeds[1:]:
            await interaction.followup.send(embed=embed)

    return workflows


def profile_command(bot):
    """Create the profile command for viewing personal statistics."""

    @app_commands.command(
        name="profile",
        description="Show your generation stats and sponsorship status",
    )
    async def profile(interaction: discord.Interaction) -> None:
        user_id = str(interaction.user.id)
        bot._reset_counts_if_needed()
        stats = bot.get_user_generation_summary(user_id)

        tier = await bot._determine_user_tier(interaction)
        listed_donor = user_id in bot.donor_users
        has_unlimited = tier.daily_limit is None

        status_details = [f"Tier: **{tier.name}**"]
        if tier.queue_priority >= 30:
            status_details.append("Priority queue access")
        if tier.max_parallel_generations > 1:
            status_details.append(f"Queue up to {tier.max_parallel_generations} at once")
        if listed_donor and tier.level < bot.donor_tier.level:
            status_details.append("Listed as donor")

        sponsorship_status = "\n".join(status_details)

        embed = discord.Embed(
            title="👤 User Profile",
            description=f"Stats for {interaction.user.mention}",
            color=0x5865F2,
        )

        if interaction.user.display_avatar:
            embed.set_thumbnail(url=interaction.user.display_avatar.url)

        stats_lines = [
            f"Today: **{stats['day']}**",
            f"7 days: **{stats['week']}**",
            f"30 days: **{stats['month']}**",
            f"All time: **{stats['total']}**",
        ]
        embed.add_field(name="📈 Generations", value="\n".join(stats_lines), inline=False)
        embed.add_field(name="💖 Sponsorship", value=sponsorship_status, inline=False)

        if has_unlimited:
            limit_lines = ["Unlimited — thank you for supporting us!"]
            limit_lines.append(f"Queue slots: **{tier.max_parallel_generations}**")
            if tier.queue_priority >= 30:
                limit_lines.append("Priority over lower tiers.")
            embed.add_field(name="💎 Access", value="\n".join(limit_lines), inline=False)
        else:
            used = int(bot.user_generation_counts.get(user_id, 0))
            limit = int(tier.daily_limit or 0)
            remaining = max(0, limit - used)
            reset_hint = bot._format_time_remaining()
            limit_lines = [
                f"Used: **{used}** / **{limit}**",
                f"Remaining today: **{remaining}**",
                f"Resets in: {reset_hint}",
            ]
            embed.add_field(name="🔒 Daily limit", value="\n".join(limit_lines), inline=False)

        embed.set_footer(text="Support us ❤️ boosty.to/rindex")

        await interaction.response.send_message(embed=embed, ephemeral=True)

    return profile
