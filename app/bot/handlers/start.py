from aiogram import Router
from aiogram.filters import CommandStart
from aiogram.types import Message
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repository import get_or_create_user

router = Router()


def _escape_markdown_v2(text: str) -> str:
    specials = r"\_*[]()~`>#+-=|{}.!"
    return "".join("\\" + ch if ch in specials else ch for ch in text)


@router.message(CommandStart())
async def start(message: Message, session: AsyncSession):
    tg_user = message.from_user
    if tg_user is None:
        return

    async with session.begin():
        await get_or_create_user(session, tg_user.id, tg_user.username)

    text = (
        "*Привет\\!* 👋\n"
        "Я — бот\\-помощник по правилам и регламентам фигурного катания\\.\n\n"
        "*Помогаю быстро находить и объяснять информацию из официальных документов:*\n\n"
        "• *ISU* \\(International Skating Union\\)\n"
        "• *ФФКР* \\(Федерация фигурного катания России\\)\n"
        "• *АРМФК* \\(локальные/методические материалы и регламенты\\)\n\n"
        "*Напиши вопрос своими словами — например:*\n"
        f"• {_escape_markdown_v2('«Какие требования к элементу?»')}\n"
        f"• {_escape_markdown_v2('«Как считается уровень?»')}\n"
        f"• {_escape_markdown_v2('«Где это прописано?»')}\n\n"
        "Я найду релевантный фрагмент и кратко объясню, что он означает\\."
    )

    await message.answer(text, parse_mode="MarkdownV2")
