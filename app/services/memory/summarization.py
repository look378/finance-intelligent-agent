"""
Summarization memory strategy.

Periodically summarizes old messages to retain important context.
"""
from typing import Optional, List

from app.services.memory.base import MemoryStrategy, MemoryContent, MessageContent
from app.services.llm.base import LLMServiceBase, LLMMessage
from app.services.llm.prompt_templates import PromptTemplates
from app.services.memory.base import MessageContent
from app.repositories.message_repository import MessageRepository


class SummarizationMemory(MemoryStrategy):
    """
    Summarization-based memory strategy.

    Periodically summarizes old messages to retain important information.
    Pros: Retains important information from long conversations
    Cons: More expensive, may lose some detail

    Attributes:
        message_repo: Message repository instance
        llm_service: LLM service for generating summaries
        summary_threshold: Number of messages before summarization
        summary_interval: Messages between each summary
    """

    def __init__(
        self,
        message_repo: MessageRepository,
        llm_service: LLMServiceBase,
        summary_threshold: int = 20,
        summary_interval: int = 10,
    ) -> None:
        """
        Initialize summarization memory.

        Args:
            message_repo: Message repository instance
            llm_service: LLM service for generating summaries
            summary_threshold: Messages before creating first summary
            summary_interval: Messages between subsequent summaries
        """
        super().__init__(message_repo)
        self.llm_service = llm_service
        self.summary_threshold = summary_threshold
        self.summary_interval = summary_interval

    async def get_context(
        self,
        session_id: int,
        max_tokens: Optional[int] = None,
    ) -> List[MessageContent]:
        """
        Retrieve context with latest summary and recent messages.

        Args:
            session_id: Chat session ID
            max_tokens: Optional maximum tokens to include

        Returns:
            List[MessageContent]: List of messages for context
        """
        # Get latest summary
        summary_msg = await self.message_repo.get_latest_summary(session_id)

        # Get recent messages
        recent = await self.message_repo.get_recent_messages(
            session_id=session_id,
            limit=self.summary_interval,
        )

        # Convert to MessageContent format
        context = []

        # Add summary as system message if exists
        if summary_msg:
            context.append(
                MessageContent(
                    role="system",
                    content=f"Previous conversation: {summary_msg.content}",
                    timestamp=summary_msg.created_at,
                )
            )

        # Add recent messages
        for msg in recent:
            context.append(
                MessageContent(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.created_at,
                )
            )

        # Truncate by tokens if needed
        if max_tokens and context:
            context = await self.truncate_by_tokens(context, max_tokens)

        return context

    async def add_message(
        self,
        session_id: int,
        message: MessageContent,
    ) -> None:
        """
        Add a message and trigger summarization if threshold reached.

        Args:
            session_id: Chat session ID
            message: Message to add
        """
        # Store message in database
        from app.models.database.message import Message
        from app.models.enums.message import MessageRole, MessageStatus

        db_message = Message(
            session_id=session_id,
            role=MessageRole(message.role),
            content=message.content,
            status=MessageStatus.COMPLETED,
        )

        await self.message_repo.create(db_message)

        # Check if we need to create a summary
        count = await self.message_repo.count_messages(session_id)

        # Create summary at threshold, then every interval
        if count == self.summary_threshold or (
            count > self.summary_threshold and
            (count - self.summary_threshold) % self.summary_interval == 0
        ):
            await self._create_summary(session_id)

    async def clear_session(self, session_id: int) -> None:
        """
        Clear all messages for a session.

        Args:
            session_id: Chat session ID
        """
        await self.message_repo.delete_by_session(session_id)

    async def _create_summary(self, session_id: int) -> None:
        """
        Create a summary of old messages.

        Args:
            session_id: Chat session ID
        """
        # Get messages to summarize (all messages before last summary_interval)
        messages = await self.message_repo.get_messages_before_summary(
            session_id=session_id,
            limit=self.summary_threshold,
        )

        if not messages:
            return

        # Build summary prompt
        conversation_text = "\n".join(
            f"{msg.role}: {msg.content}"
            for msg in messages
        )

        prompt = PromptTemplates.get_summarization_prompt(
            text=conversation_text,
            max_length=500
        )

        # Generate summary using LLM
        response = await self.llm_service.generate(
            messages=[LLMMessage(role="user", content=prompt)],
            max_tokens=300,
            temperature=0.3,
        )

        # Store summary as a system message
        from app.models.database.message import Message
        from app.models.enums.message import MessageRole, MessageStatus

        summary_message = Message(
            session_id=session_id,
            role=MessageRole.SYSTEM,
            content=response.content,
            status=MessageStatus.COMPLETED,
        )

        await self.message_repo.create_summary(summary_message)

        # Archive old messages
        await self.message_repo.archive_messages(
            session_id=session_id,
            count=len(messages)
        )
