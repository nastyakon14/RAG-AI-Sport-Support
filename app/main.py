import asyncio
import logging

from aiogram import Bot, Dispatcher

from app.bot.middlewares.db import DbSessionMiddleware
from app.bot.routers import setup_routers
from app.config import settings


async def main():
    logging.basicConfig(level=logging.INFO)

    bot = Bot(token=settings.bot_token)
    dp = Dispatcher()

    dp.update.middleware(DbSessionMiddleware())
    dp.include_router(setup_routers())

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
