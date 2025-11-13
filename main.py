import asyncio
import os
import sys
from logger import logger
from src.bot.imagesmith import ComfyUIBot

# На Windows нужно явно задать политику цикла событий
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

async def main():
    bot = ComfyUIBot()

    logger.info("Starting bot...")
    try:
        discord_token = os.getenv("DISCORD_TOKEN") or bot.workflow_manager.config["discord"]["token"]
        if not discord_token:
            raise RuntimeError("Discord token is missing. Set DISCORD_TOKEN env var or fill configuration.yml.")

        await bot.start(discord_token)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await bot.cleanup()

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        try:
            await bot.cleanup()
        finally:
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
