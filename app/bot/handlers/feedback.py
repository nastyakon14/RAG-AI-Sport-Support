from aiogram import F, Router
from aiogram.types import CallbackQuery
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repository import upsert_feedback

router = Router()


@router.callback_query(F.data.startswith("fb:"))
async def on_feedback(call: CallbackQuery, session: AsyncSession):
    parts = (call.data or "").split(":")
    if len(parts) != 3:
        await call.answer("Некорректные данные", show_alert=True)
        return

    _, like_str, response_id_str = parts
    is_like = like_str == "1"

    try:
        response_id = int(response_id_str)
    except ValueError:
        await call.answer("Некорректный id", show_alert=True)
        return

    async with session.begin():
        await upsert_feedback(session, response_id=response_id, is_like=is_like)

    await call.answer("Спасибо за фидбек!")
