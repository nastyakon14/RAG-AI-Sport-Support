from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, ForeignKey, Identity, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    username: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    messages: Mapped[list["Message"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    text: Mapped[str] = mapped_column(String(4096), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    user: Mapped["User"] = relationship(back_populates="messages")

    response: Mapped["Response | None"] = relationship(
        back_populates="message",
        uselist=False,
        cascade="all, delete-orphan",
        single_parent=True,
    )


class Response(Base):
    __tablename__ = "responses"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    message_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("messages.id", ondelete="CASCADE"),
        index=True,
        unique=True,
        nullable=False,
    )
    text: Mapped[str] = mapped_column(String(8192), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    message: Mapped["Message"] = relationship(back_populates="response")

    feedback: Mapped["Feedback | None"] = relationship(
        back_populates="response",
        uselist=False,
        cascade="all, delete-orphan",
        single_parent=True,
    )


class Feedback(Base):
    __tablename__ = "feedbacks"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    response_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("responses.id", ondelete="CASCADE"),
        index=True,
        unique=True,
        nullable=False,
    )
    is_like: Mapped[bool] = mapped_column(nullable=False)
    comment: Mapped[str | None] = mapped_column(String(2048), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    response: Mapped["Response"] = relationship(back_populates="feedback")
