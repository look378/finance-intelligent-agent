"""Chat session database model"""
from typing import List

from sqlalchemy import ForeignKey, String, Integer, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.database.base import Base, TimestampMixin


class ChatSession(Base, TimestampMixin):
    """
    Chat session model representing conversation sessions.

    Attributes:
        id: Primary key
        user_id: Foreign key to user
        title: Session title
        memory_type: Type of memory strategy to use
        context_window: Number of messages to keep in context
        metadata: Additional metadata as JSON
        user: Relationship to User
        messages: Relationship to Message objects
    """
    __tablename__ = "chat_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(
        String(255),
        default="New Chat",
        nullable=False,
    )
    memory_type: Mapped[str] = mapped_column(
        String(50),
        default="sliding_window",
        nullable=False,
    )
    context_window: Mapped[int] = mapped_column(
        Integer,
        default=10,
        nullable=False,
    )
    session_metadata: Mapped[dict | None] = mapped_column(JSON, default=None)

    # Relationships
    user: Mapped["User"] = relationship(
        back_populates="sessions",
    )
    messages: Mapped[List["Message"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    def __repr__(self) -> str:
        return f"<ChatSession(id={self.id}, title={self.title}, user_id={self.user_id})>"
