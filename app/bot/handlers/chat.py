from aiogram import Router
from aiogram.types import Message
from sqlalchemy.ext.asyncio import AsyncSession

from app.bot.keyboards import feedback_kb
from app.db.repository import create_message, create_response, get_or_create_user

router = Router()


@router.message()
async def on_message(message: Message, session: AsyncSession):
    tg_user = message.from_user
    if tg_user is None or message.text is None:
        return

    user_text = message.text

    system_text = f"Ты написал: {user_text}"

    async with session.begin():
        await get_or_create_user(session, tg_user.id, tg_user.username)
        msg = await create_message(session, tg_user.id, user_text)
        resp = await create_response(session, msg.id, system_text)

    await message.answer(system_text, reply_markup=feedback_kb(resp.id))
