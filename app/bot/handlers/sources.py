from aiogram import F, Router
from aiogram.types import CallbackQuery
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Response
from app.utils.rag_text import split_answer_and_links

router = Router()


@router.callback_query(F.data.startswith("src:"))
async def on_sources(call: CallbackQuery, session: AsyncSession):
    _, resp_id_str = (call.data or "").split(":", 1)

    try:
        response_id = int(resp_id_str)
    except ValueError:
        await call.answer("Некорректный id", show_alert=True)
        return

    resp = await session.get(Response, response_id)
    if resp is None:
        await call.answer("Ответ не найден", show_alert=True)
        return

    _, links = split_answer_and_links(resp.text)

    if not links:
        await call.answer("Ссылок нет", show_alert=True)
        return

    await call.message.answer(
        links,
        parse_mode="HTML",
        disable_web_page_preview=True,
    )
    await call.answer()
