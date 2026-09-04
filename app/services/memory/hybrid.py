"""
Hybrid memory strategy combining sliding window and summarization.

Automatically switches strategies based on conversation length.
"""
from typing import Optional, List

from app.services.memory.base import MemoryStrategy, MemoryContent, MessageContent
from app.services.memory.sliding_window import SlidingWindowMemory
from app.services.memory.summarization import SummarizationMemory


class HybridMemory(MemoryStrategy):
    """
    Hybrid memory strategy combining sliding window and summarization.

    Automatically switches strategies based on conversation length:
    - Short conversations (< hybrid_threshold): Use sliding window
    - Long conversations (>= hybrid_threshold): Use summarization

    This provides optimal performance and memory retention.
    """

    def __init__(
        self,
        sliding_window: SlidingWindowMemory,
        summarization: SummarizationMemory,
        hybrid_threshold: int = 30,
    ) -> None:
        """
        Initialize hybrid memory strategy.

        Args:
            sliding_window: Sliding window memory instance
            summarization: Summarization memory instance
            hybrid_threshold: Messages threshold for switching to summarization
        """
        # Don't call super().__init__ as we delegate to strategies
        self.sliding_window = sliding_window
        self.summarization = summarization
        self.hybrid_threshold = hybrid_threshold

    async def get_context(
        self,
        session_id: int,
        max_tokens: Optional[int] = None,
    ) -> List[MessageContent]:
        """
        Retrieve context using the appropriate strategy.

        Args:
            session_id: Chat session ID
            max_tokens: Optional maximum tokens to include

        Returns:
            List[MessageContent]: List of messages for context
        """
        # Check message count to determine strategy
        count = await self.sliding_window.message_repo.count_messages(session_id)

        if count < self.hybrid_threshold:
            # Use sliding window for short conversations
            return await self.sliding_window.get_context(session_id, max_tokens)
        else:
            # Use summarization for long conversations
            return await self.summarization.get_context(session_id, max_tokens)

    async def add_message(
        self,
        session_id: int,
        message: MessageContent,
    ) -> None:
        """
        Add a message using the appropriate strategy.

        Args:
            session_id: Chat session ID
            message: Message to add
        """
        # Check message count to determine strategy
        count = await self.sliding_window.message_repo.count_messages(session_id)

        if count < self.hybrid_threshold:
            # Use sliding window logic (no summarization trigger)
            await self.sliding_window.add_message(session_id, message)
        else:
            # Use summarization logic (includes auto-summarization)
            await self.summarization.add_message(session_id, message)

    async def clear_session(self, session_id: int) -> None:
        """
        Clear all messages for a session.

        Args:
            session_id: Chat session ID
        """
        # Either strategy works - use sliding_window for simplicity
        await self.sliding_window.message_repo.delete_by_session(session_id)

    async def estimate_tokens(self, messages: List[MessageContent]) -> int:
        """
        Estimate the number of tokens in messages.

        Args:
            messages: List of messages

        Returns:
            int: Estimated token count
        """
        return await self.sliding_window.estimate_tokens(messages)

    async def truncate_by_tokens(
        self,
        messages: List[MessageContent],
        max_tokens: int,
    ) -> List[MessageContent]:
        """
        Truncate messages to fit within token limit.

        Args:
            messages: List of messages (newest last)
            max_tokens: Maximum tokens allowed

        Returns:
            List[MessageContent]: Truncated list of messages
        """
        return await self.sliding_window.truncate_by_tokens(messages, max_tokens)
