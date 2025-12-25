import asyncio

from aiogram import Router
from aiogram.types import Message
from sqlalchemy.ext.asyncio import AsyncSession

from app.bot.keyboards import feedback_kb
from app.db.repository import create_message, create_response, get_or_create_user
from app.rag.pipeline import RagFS
from app.utils.rag_text import split_answer_and_links

router = Router()


@router.message()
async def on_message(message: Message, session: AsyncSession, rag_fs: RagFS):
    tg_user = message.from_user
    if tg_user is None or message.text is None:
        return

    user_text = message.text

    wait_msg = await message.answer(
        "⏳ Пожалуйста, подождите — выполняется поиск информации по вашему запросу…"
    )

    try:
        system_text = await asyncio.to_thread(rag_fs.find_answer, user_text)

        answer_text, links_block = split_answer_and_links(system_text)

        async with session.begin():
            await get_or_create_user(session, tg_user.id, tg_user.username)
            msg = await create_message(session, tg_user.id, user_text)
            resp = await create_response(session, msg.id, system_text)
        try:
            await wait_msg.delete()
        except Exception:
            pass

        await message.answer(
            answer_text if answer_text else "Ответ сформирован.",
            reply_markup=feedback_kb(resp.id),
            parse_mode="HTML",
        )

    finally:
        try:
            await wait_msg.delete()
        except Exception:
            pass
