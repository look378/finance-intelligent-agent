"""
Sliding window memory strategy.

Keeps the last N messages in conversation context.
Simple, fast, and predictable.
"""
from typing import Optional, List

from app.services.memory.base import MemoryStrategy, MemoryContent, MessageContent
from app.services.memory.base import MessageContent
from app.repositories.message_repository import MessageRepository


class SlidingWindowMemory(MemoryStrategy):
    """
    Sliding window memory strategy.

    Keeps the last N messages in conversation context.
    Pros: Simple, fast, predictable
    Cons: May lose important context from earlier messages

    Attributes:
        message_repo: Message repository instance
        window_size: Number of messages to keep (default: 10)
    """

    def __init__(
        self,
        message_repo: MessageRepository,
        window_size: int = 10,
    ) -> None:
        """
        Initialize sliding window memory.

        Args:
            message_repo: Message repository instance
            window_size: Number of messages to keep in context
        """
        super().__init__(message_repo)
        self.window_size = window_size

    async def get_context(
        self,
        session_id: int,
        max_tokens: Optional[int] = None,
    ) -> List[MessageContent]:
        """
        Retrieve recent messages within the window size.

        Args:
            session_id: Chat session ID
            max_tokens: Optional maximum tokens to include

        Returns:
            List[MessageContent]: List of messages for context
        """
        # Get recent messages from database
        messages = await self.message_repo.get_recent_messages(
            session_id=session_id,
            limit=self.window_size,
        )

        # Convert to MessageContent format
        context = [
            MessageContent(
                role=msg.role,
                content=msg.content,
                timestamp=msg.created_at,
            )
            for msg in messages
        ]

        # Truncate by tokens if max_tokens specified
        if max_tokens and context:
            context = await self.truncate_by_tokens(context, max_tokens)

        return context

    async def add_message(
        self,
        session_id: int,
        message: MessageContent,
    ) -> None:
        """
        Add a message to the session (stored in database).

        Args:
            session_id: Chat session ID
            message: Message to add
        """
        # Create message in database
        from app.models.database.message import Message
        from app.models.enums.message import MessageRole, MessageStatus

        db_message = Message(
            session_id=session_id,
            role=MessageRole(message["role"]),
            content=message["content"],
            status=MessageStatus.COMPLETED,
        )

        await self.message_repo.create(db_message)

    async def clear_session(self, session_id: int) -> None:
        """
        Clear all messages for a session.

        Args:
            session_id: Chat session ID
        """
        await self.message_repo.delete_by_session(session_id)
