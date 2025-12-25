from typing import Any, Awaitable, Callable, Dict

from aiogram import BaseMiddleware
from aiogram.types import TelegramObject
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import SessionMaker


class DbSessionMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any],
    ) -> Any:
        session: AsyncSession = SessionMaker()
        try:
            data["session"] = session
            result = await handler(event, data)
            return result
        finally:
            await session.close()
