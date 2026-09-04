"""
Session service for chat session management.

Handles session creation, retrieval, updating, and deletion.
"""
from typing import Optional, List

from app.repositories.session_repository import SessionRepository
from app.models.database.session import ChatSession
from app.core.exceptions import NotFoundError, ValidationError


class SessionService:
    """
    Service for managing chat sessions.

    Provides methods for session CRUD operations with
    proper authorization checks.
    """

    def __init__(self, session_repository: SessionRepository) -> None:
        """
        Initialize the session service.

        Args:
            session_repository: Session repository instance
        """
        self.session_repository = session_repository

    async def create_session(
        self,
        user_id: int,
        title: Optional[str] = None,
        memory_type: str = "sliding_window",
        context_window: int = 10,
    ) -> ChatSession:
        """
        Create a new chat session for a user.

        Args:
            user_id: User ID
            title: Optional session title
            memory_type: Memory strategy type
            context_window: Number of messages to keep in context

        Returns:
            ChatSession: Created session

        Raises:
            ValidationError: If memory_type or context_window is invalid
        """
        # Validate memory type
        valid_memory_types = ["sliding_window", "summarization", "hybrid"]
        if memory_type not in valid_memory_types:
            raise ValidationError(
                f"Invalid memory_type: {memory_type}. "
                f"Must be one of {valid_memory_types}"
            )

        # Validate context window
        if context_window < 1 or context_window > 100:
            raise ValidationError("context_window must be between 1 and 100")

        # Create session
        session = ChatSession(
            user_id=user_id,
            title=title or "New Chat",
            memory_type=memory_type,
            context_window=context_window,
        )

        created_session = await self.session_repository.create(session)
        return created_session

    async def get_user_sessions(
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
        return await self.session_repository.get_by_user_id(user_id, skip, limit)

    async def get_session_by_id(
        self,
        session_id: int,
        user_id: int,
    ) -> ChatSession:
        """
        Get a session by ID with authorization check.

        Args:
            session_id: Session ID
            user_id: User ID (for authorization)

        Returns:
            ChatSession: Session if found and user owns it

        Raises:
            NotFoundError: If session not found or user doesn't own it
        """
        session = await self.session_repository.get_by_id(session_id, ChatSession)

        if not session:
            raise NotFoundError("Session", str(session_id))

        if session.user_id != user_id:
            raise NotFoundError("Session", str(session_id))

        return session

    async def update_session(
        self,
        session_id: int,
        user_id: int,
        title: Optional[str] = None,
        memory_type: Optional[str] = None,
        context_window: Optional[int] = None,
    ) -> ChatSession:
        """
        Update a session.

        Args:
            session_id: Session ID
            user_id: User ID (for authorization)
            title: New title
            memory_type: New memory type
            context_window: New context window size

        Returns:
            ChatSession: Updated session

        Raises:
            NotFoundError: If session not found or user doesn't own it
            ValidationError: If parameters are invalid
        """
        # Get session with authorization check
        session = await self.get_session_by_id(session_id, user_id)

        # Update fields if provided
        if title is not None:
            session.title = title

        if memory_type is not None:
            valid_memory_types = ["sliding_window", "summarization", "hybrid"]
            if memory_type not in valid_memory_types:
                raise ValidationError(
                    f"Invalid memory_type: {memory_type}. "
                    f"Must be one of {valid_memory_types}"
                )
            session.memory_type = memory_type

        if context_window is not None:
            if context_window < 1 or context_window > 100:
                raise ValidationError("context_window must be between 1 and 100")
            session.context_window = context_window

        # Save updates
        updated_session = await self.session_repository.update(session)
        return updated_session

    async def delete_session(
        self,
        session_id: int,
        user_id: int,
    ) -> None:
        """
        Delete a session.

        Args:
            session_id: Session ID
            user_id: User ID (for authorization)

        Raises:
            NotFoundError: If session not found or user doesn't own it
        """
        # Get session with authorization check
        session = await self.get_session_by_id(session_id, user_id)

        # Delete session (cascade will delete associated messages)
        await self.session_repository.delete(session)

    async def count_user_sessions(self, user_id: int) -> int:
        """
        Count sessions for a user.

        Args:
            user_id: User ID

        Returns:
            int: Number of sessions
        """
        return await self.session_repository.count_by_user_id(user_id)
