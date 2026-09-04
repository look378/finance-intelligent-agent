"""Message database model"""
from sqlalchemy import ForeignKey, String, Text, Integer, JSON, Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.database.base import Base, TimestampMixin
from app.models.enums.message import MessageRole, MessageStatus


class Message(Base, TimestampMixin):
    """
    Message model representing chat messages in a session.

    Attributes:
        id: Primary key
        session_id: Foreign key to chat session
        role: Message role (user/assistant/system)
        content: Message content
        intent: Detected intent (for user messages)
        status: Processing status
        token_count: Number of tokens in message
        metadata: Additional metadata (sources, etc.)
        session: Relationship to ChatSession
    """
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[MessageRole] = mapped_column(
        SQLEnum(MessageRole),
        nullable=False,
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    intent: Mapped[str | None] = mapped_column(
        String(100),
        index=True,
        default=None,
    )
    status: Mapped[MessageStatus] = mapped_column(
        SQLEnum(MessageStatus),
        default=MessageStatus.COMPLETED,
        nullable=False,
    )
    token_count: Mapped[int | None] = mapped_column(Integer, default=None)
    message_metadata: Mapped[dict | None] = mapped_column(JSON, default=None)
    user_rating: Mapped[int | None] = mapped_column(Integer, default=None)
    feedback_text: Mapped[str | None] = mapped_column(Text, default=None)

    # Relationships
    session: Mapped["ChatSession"] = relationship(
        back_populates="messages",
    )

    def __repr__(self) -> str:
        return f"<Message(id={self.id}, role={self.role}, session_id={self.session_id})>"
