from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Feedback, Message, Response, User


async def get_or_create_user(
    session: AsyncSession, user_id: int, username: str | None
) -> type[User] | User:
    user = await session.get(User, user_id)
    if user:
        if username is not None and user.username != username:
            user.username = username
        return user

    user = User(id=user_id, username=username)
    session.add(user)
    return user


async def create_message(session: AsyncSession, user_id: int, text: str) -> Message:
    msg = Message(user_id=user_id, text=text)
    session.add(msg)
    await session.flush()
    return msg


async def create_response(
    session: AsyncSession, message_id: int, text: str
) -> Response:
    resp = Response(message_id=message_id, text=text)
    session.add(resp)
    await session.flush()
    return resp


async def upsert_feedback(
    session: AsyncSession, response_id: int, is_like: bool, comment: str | None = None
) -> Feedback:
    q = select(Feedback).where(Feedback.response_id == response_id)
    fb = (await session.execute(q)).scalar_one_or_none()

    if fb is None:
        fb = Feedback(response_id=response_id, is_like=is_like, comment=comment)
        session.add(fb)
        await session.flush()
        return fb

    fb.is_like = is_like
    fb.comment = comment
    return fb
