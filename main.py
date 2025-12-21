import asyncio
import sys

from logger import logger
from src.discord_bot import DiscordBot

# On Windows explicitly set selector policy
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def main() -> None:
    bot = DiscordBot()
    logger.info("Starting bot...")
    try:
        discord_token = bot.config_manager.discord_token()
        await bot.start(discord_token)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        await bot.close()
    except Exception as exc:
        logger.error(f"Fatal error: {exc}", exc_info=True)
        try:
            await bot.close()
        finally:
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
