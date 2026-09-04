"""
Session repository for chat session data access.

Provides database operations specific to the ChatSession model.
"""
from typing import List

from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database.session import ChatSession
from app.repositories.base import BaseRepository


class SessionRepository(BaseRepository[ChatSession]):
    """
    Repository for ChatSession entity operations.

    Extends BaseRepository with session-specific queries.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize the session repository.

        Args:
            session: Async database session
        """
        super().__init__(session)

    async def get_by_user_id(
        self,
        user_id: int,
        skip: int = 0,
        limit: int = 100,
    ) -> List[ChatSession]:
        """
        Get all sessions for a user with pagination.

        Args:
            user_id: User ID
            skip: Number of sessions to skip
            limit: Maximum number of sessions to return

        Returns:
            List[ChatSession]: List of user's sessions
        """
        stmt = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_with_messages(self, session_id: int) -> ChatSession | None:
        """
        Get a session with messages preloaded.

        Args:
            session_id: Session ID

        Returns:
            ChatSession | None: Session with messages if found, None otherwise
        """
        stmt = (
            select(ChatSession)
            .where(ChatSession.id == session_id)
            .options(selectinload(ChatSession.messages))
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def count_by_user_id(self, user_id: int) -> int:
        """
        Count sessions for a user.

        Args:
            user_id: User ID

        Returns:
            int: Number of sessions
        """
        stmt = select(func.count()).select_from(ChatSession).where(ChatSession.user_id == user_id)
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def get_latest_session(self, user_id: int) -> ChatSession | None:
        """
        Get the most recent session for a user.

        Args:
            user_id: User ID

        Returns:
            ChatSession | None: Latest session if found, None otherwise
        """
        stmt = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
            .limit(1)
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
