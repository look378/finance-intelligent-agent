"""
User repository for user data access.

Provides database operations specific to the User model.
"""
from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.database.user import User
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    """
    Repository for User entity operations.

    Extends BaseRepository with user-specific queries.
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize the user repository.

        Args:
            session: Async database session
        """
        super().__init__(session)

    async def get_by_email(self, email: str) -> User | None:
        """
        Get a user by email address (case-insensitive).

        Args:
            email: User email address

        Returns:
            User | None: User if found, None otherwise
        """
        stmt = select(User).where(User.email.ilike(email))
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def exists_by_email(self, email: str) -> bool:
        """
        Check if a user exists by email address.

        Args:
            email: Email address to check

        Returns:
            bool: True if user exists, False otherwise
        """
        return await self.exists(User.email.ilike(email))

    async def list_users(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[User]:
        """
        List all users with pagination.

        Args:
            skip: Number of users to skip
            limit: Maximum number of users to return

        Returns:
            List[User]: List of users
        """
        stmt = (
            select(User)
            .order_by(User.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_with_sessions(self, user_id: int) -> User | None:
        """
        Get a user with their sessions preloaded.

        Args:
            user_id: User ID

        Returns:
            User | None: User with sessions if found, None otherwise
        """
        stmt = (
            select(User)
            .where(User.id == user_id)
            .options(selectinload(User.sessions))
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
