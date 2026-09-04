"""
Message repository for message data access.

Provides database operations specific to the Message model.
"""
from typing import List

from sqlalchemy import select, desc
from sqlalchemy.orm import selectinload

from app.models.database.message import Message
from app.repositories.base import BaseRepository


class MessageRepository(BaseRepository[Message]):
    """
    Repository for Message entity operations.

    Extends BaseRepository with message-specific queries.
    """

    def __init__(self, session) -> None:
        """
        Initialize the message repository.

        Args:
            session: Async database session
        """
        super().__init__(session)

    async def get_recent_messages(
        self,
        session_id: int,
        limit: int = 50,
    ) -> List[Message]:
        """
        Get recent messages for a session.

        Args:
            session_id: Session ID
            limit: Maximum number of messages to return

        Returns:
            List[Message]: List of recent messages ordered by creation time
        """
        stmt = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_session(
        self,
        session_id: int,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Message]:
        """
        Get all messages for a session with pagination.

        Args:
            session_id: Session ID
            skip: Number of messages to skip
            limit: Maximum number of messages to return

        Returns:
            List[Message]: List of messages
        """
        stmt = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.asc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_latest_summary(
        self,
        session_id: int,
    ) -> Message | None:
        """
        Get the latest summary message for a session.

        Args:
            session_id: Session ID

        Returns:
            Message | None: Latest summary message if found
        """
        stmt = (
            select(Message)
            .where(
                Message.session_id == session_id,
                Message.role == "system"
            )
            .order_by(Message.created_at.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_messages_before_summary(
        self,
        session_id: int,
        limit: int = 20,
    ) -> List[Message]:
        """
        Get messages before the latest summary.

        Args:
            session_id: Session ID
            limit: Maximum number of messages to return

        Returns:
            List[Message]: Messages before the latest summary
        """
        # Get latest summary first
        summary = await self.get_latest_summary(session_id)

        if not summary:
            # No summary yet, return recent messages
            return await self.get_recent_messages(session_id, limit)

        # Get messages created before the summary
        stmt = (
            select(Message)
            .where(
                Message.session_id == session_id,
                Message.created_at < summary.created_at
            )
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def create_summary(
        self,
        message: Message,
    ) -> Message:
        """
        Create a summary message in the database.

        Args:
            message: Message instance to create

        Returns:
            Message: Created message
        """
        return await self.create(message)

    async def archive_messages(
        self,
        session_id: int,
        count: int,
    ) -> None:
        """
        Archive (soft delete) old messages after summarization.

        Args:
            session_id: Session ID
            count: Number of old messages to archive
        """
        # Get old messages
        messages = await self.get_messages_before_summary(session_id, limit=count)

        # Mark as archived (you could use a separate table for this)
        for message in messages:
            message.message_metadata = message.message_metadata or {}
            message.message_metadata["archived"] = True
            await self.update(message)

    async def delete_by_session(
        self,
        session_id: int,
    ) -> None:
        """
        Delete all messages for a session.

        Args:
            session_id: Session ID
        """
        # Note: This will cascade delete due to foreign key constraint
        stmt = select(Message).where(Message.session_id == session_id)
        result = await self.session.execute(stmt)

        for message in result.scalars().all():
            await self.delete(message)

    async def count_messages(
        self,
        session_id: int,
    ) -> int:
        """
        Count all messages for a session.

        Args:
            session_id: Session ID

        Returns:
            int: Number of messages
        """
        from sqlalchemy import func

        stmt = select(func.count()).select_from(Message).where(
            Message.session_id == session_id
        )
        result = await self.session.execute(stmt)
        return result.scalar() or 0
