# Hentai-Diffusion Discord Bot

Discord bot for orchestrating ComfyUI generations with a structured, testable architecture.

## Features (behavior preserved)
- Slash commands: `/rgen`, `/reforge`, `/upscale`, `/workflows`, `/limits`, `/spoiler list|add|remove`, `/profile`.
- Priority queue with role-based limits, parallelism, and cross-bot sync.
- Prompt/model/LoRA presets, resolution presets, workflow settings hooks.
- Cancellation, reuse of last request, spoiler tagging, access-guild enforcement.
- Multi-instance ComfyUI client with retries, reconnect, and timeout monitoring.

## Project layout
- `src/discord_bot.py` — Discord-facing bot wiring commands to orchestrator.
- `src/app/` — config handling, generation orchestration, sync, models, storage.
- `src/comfy/` — ComfyUI client and workflow preparation.
- `src/core/` — queue, hooks, security, plugin interfaces.
- `src/ui/` — embeds/views/buttons.
- `configuration.yml` — main config (Discord token, ComfyUI instances, presets, workflows, security).

## Requirements
Python 3.11+ with `pip install -r requirements.txt`.

## Configuration
1. Copy `configuration.yml` and fill:
   - `discord.token` or set env `DISCORD_TOKEN`.
   - `comfyui.instances` with your ComfyUI endpoint(s); `input_dir` points to ComfyUI input folder.
   - Optional: `security.access_guild_id`, role IDs, spoiler tags, workflow settings.
2. Preset files are referenced in `configuration.yml` (`data/prompt_presets.json`, etc.).
3. (Optional) Sync multiple bots via hidden channel: set `sync.channel_id`, `sync.shared_secret`, `sync.prefix`.

## Running
```bash
python main.py
```
On Windows, `start_bot.bat` can be used after installing dependencies.

## Development & Testing
Install dev deps from `requirements.txt`, then:
```bash
pytest
```

## Quick self-check
1. Start bot with valid `DISCORD_TOKEN` and ComfyUI running.
2. In Discord:
   - `/rgen prompt:<text>` → enqueues generation, shows queue position and progress; buttons Cancel/Reuse work.
   - `/reforge` or `/upscale` with an image attachment enforce image validation.
   - `/workflows` lists workflows (landscape variants hidden as before).
   - `/limits` and `/profile` show tier, usage bars, and stats.
   - `/spoiler add/list/remove` edits spoiler tags (Manage Server required).
3. Cross-bot sync: active slot/limit updates appear in the hidden sync channel if configured.
