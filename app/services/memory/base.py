"""
Base interface for memory management strategies.

Provides abstract classes for conversation memory management.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from app.models.database.message import Message


# Type alias for MessageContent - could be a Message object or dict
MessageContent = dict


@dataclass
class MemoryContent:
    """
    Content stored in memory.

    Attributes:
        messages: List of messages in memory
        summary: Optional summary of the content
        metadata: Optional additional metadata
    """
    messages: List[Message] = field(default_factory=list)
    summary: Optional[str] = None
    metadata: Optional[dict] = None


class MemoryStrategy(ABC):
    """
    Abstract base class for memory management strategies.

    All memory strategy implementations must inherit from this class
    and implement the required methods.
    """

    def __init__(self, message_repo) -> None:
        """
        Initialize the memory strategy.

        Args:
            message_repo: Message repository for database operations
        """
        self.message_repo = message_repo

    @abstractmethod
    async def get_context(
        self,
        session_id: int,
        max_tokens: Optional[int] = None,
    ) -> List[MessageContent]:
        """
        Retrieve relevant context for the session.

        Args:
            session_id: Chat session ID
            max_tokens: Optional maximum tokens to include

        Returns:
            List[MessageContent]: List of messages for context

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("get_context() must be implemented by subclass")

    @abstractmethod
    async def add_message(
        self,
        session_id: int,
        message: MessageContent,
    ) -> None:
        """
        Add a message to the session memory.

        Args:
            session_id: Chat session ID
            message: Message to add

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("add_message() must be implemented by subclass")

    @abstractmethod
    async def clear_session(self, session_id: int) -> None:
        """
        Clear all messages for a session.

        Args:
            session_id: Chat session ID

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("clear_session() must be implemented by subclass")

    async def estimate_tokens(self, messages: List[MessageContent]) -> int:
        """
        Estimate the number of tokens in messages.

        Args:
            messages: List of messages

        Returns:
            int: Estimated token count
        """
        # Rough estimate: ~4 characters per token
        return sum(len(m.content) // 4 for m in messages)

    async def truncate_by_tokens(
        self,
        messages: List[MessageContent],
        max_tokens: int,
    ) -> List[MessageContent]:
        """
        Truncate messages to fit within token limit.

        Keeps most recent messages that fit within the limit.

        Args:
            messages: List of messages (newest last)
            max_tokens: Maximum tokens allowed

        Returns:
            List[MessageContent]: Truncated list of messages
        """
        if not messages:
            return []

        # Start with newest messages and add until we hit the limit
        result = []
        total_tokens = 0

        for message in reversed(messages):
            message_tokens = await self.estimate_tokens([message])

            if total_tokens + message_tokens > max_tokens:
                break

            result.insert(0, message)
            total_tokens += message_tokens

        return result
