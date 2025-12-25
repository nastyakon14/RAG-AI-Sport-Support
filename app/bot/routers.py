from aiogram import Router

from app.bot.handlers import chat, feedback, sources, start


def setup_routers() -> Router:
    r = Router()
    r.include_router(start.router)
    r.include_router(feedback.router)
    r.include_router(sources.router)
    r.include_router(chat.router)
    return r
