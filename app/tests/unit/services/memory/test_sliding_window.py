"""Tests for sliding window memory strategy"""
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock


class TestSlidingWindowMemory:
    """Test sliding window memory strategy"""

    @pytest.mark.asyncio
    async def test_get_context_returns_recent_messages(self):
        """Test that get_context returns last N messages"""
        # Arrange
        from app.services.memory.sliding_window import SlidingWindowMemory
        from app.models.schemas.chat import MessageContent
        from app.models.database.message import Message
        from app.repositories.message_repository import MessageRepository

        mock_repo = Mock(spec=MessageRepository)
        memory = SlidingWindowMemory(message_repo=mock_repo, window_size=3)

        # Mock messages from database
        messages = [
            Message(id=1, content="Msg 1"),
            Message(id=2, content="Msg 2"),
            Message(id=3, content="Msg 3"),
            Message(id=4, content="Msg 4"),
            Message(id=5, content="Msg 5"),
        ]

        # Convert to MessageContent format
        mock_repo.get_recent_messages = AsyncMock(
            return_value=[Message(
                id=m.id,
                role="user",
                content=m.content,
                created_at=datetime.now()
            ) for m in messages]
        )

        # Act
        context = await memory.get_context(session_id=1)

        # Assert
        assert len(context) == 3  # Window size
        # Should get most recent messages
        assert context[0].content == "Msg 3"

    @pytest.mark.asyncio
    async def test_add_message_stores_in_database(self):
        """Test that add_message stores in database"""
        # Arrange
        from app.services.memory.sliding_window import SlidingWindowMemory
        from app.models.schemas.chat import MessageContent
        from app.models.database.message import Message
        from app.repositories.message_repository import MessageRepository

        mock_repo = Mock(spec=MessageRepository)
        memory = SlidingWindowMemory(message_repo=mock_repo, window_size=3)

        message = MessageContent(
            role="user",
            content="New message",
            timestamp=datetime.now()
        )

        mock_repo.create = AsyncMock()
        mock_repo.get_by_id = AsyncMock(return_value=Message(id=1))

        # Act
        await memory.add_message(session_id=1, message=message)

        # Assert
        mock_repo.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_clear_session_deletes_messages(self):
        """Test that clear_session deletes all messages"""
        # Arrange
        from app.services.memory.sliding_window import SlidingWindowMemory
        from app.repositories.message_repository import MessageRepository

        mock_repo = Mock(spec=MessageRepository)
        memory = SlidingWindowMemory(message_repo=mock_repo, window_size=3)

        mock_repo.delete_by_session = AsyncMock()

        # Act
        await memory.clear_session(session_id=1)

        # Assert
        mock_repo.delete_by_session.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_truncate_by_max_tokens(self):
        """Test truncating messages to fit max tokens"""
        # Arrange
        from app.services.memory.sliding_window import SlidingWindowMemory
        from app.repositories.message_repository import MessageRepository
        from app.models.schemas.chat import MessageContent

        mock_repo = Mock(spec=MessageRepository)
        memory = SlidingWindowMemory(message_repo=mock_repo, window_size=10)

        messages = [
            MessageContent(role="user", content=f"Message {i}", timestamp=datetime.now())
            for i in range(5)
        ]

        # Act - Only allow space for 2 messages
        truncated = await memory.truncate_by_tokens(messages, max_tokens=50)

        # Assert - Should keep most recent messages that fit
        assert len(truncated) <= 2

    @pytest.mark.asyncio
    async def test_get_context_with_max_tokens(self):
        """Test get_context respects max_tokens parameter"""
        # Arrange
        from app.services.memory.sliding_window import SlidingWindowMemory
        from app.models.database.message import Message
        from app.repositories.message_repository import MessageRepository

        mock_repo = Mock(spec=MessageRepository)
        memory = SlidingWindowMemory(message_repo=mock_repo, window_size=10)

        # Mock 5 messages, each ~10 tokens
        messages = [
            Message(id=i, role="user", content=f"Message number {i}")
            for i in range(1, 6)
        ]

        mock_repo.get_recent_messages = AsyncMock(
            return_value=[
                Message(
                    id=m.id,
                    role="user",
                    content=m.content,
                    created_at=datetime.now()
                ) for m in messages
            ]
        )

        # Act - Only allow 30 tokens (space for ~3 messages)
        context = await memory.get_context(session_id=1, max_tokens=30)

        # Assert
        assert len(context) <= 3

    @pytest.mark.asyncio
    async def test_estimate_tokens(self):
        """Test token estimation"""
        # Arrange
        from app.services.memory.sliding_window import SlidingWindowMemory
        from app.repositories.message_repository import MessageRepository

        mock_repo = Mock(spec=MessageRepository)
        memory = SlidingWindowMemory(message_repo=mock_repo, window_size=10)

        messages = [
            MessageContent(role="user", content="Hello world!", timestamp=datetime.now())
        ]

        # Act
        count = await memory.estimate_tokens(messages)

        # Assert
        # "Hello world!" is ~12 chars, ~3 tokens
        assert count > 0
        assert count < 10

    def test_initialization(self):
        """Test strategy initialization"""
        # Arrange
        from app.services.memory.sliding_window import SlidingWindowMemory
        from app.repositories.message_repository import MessageRepository

        mock_repo = Mock(spec=MessageRepository)

        # Act
        memory = SlidingWindowMemory(
            message_repo=mock_repo,
            window_size=15
        )

        # Assert
        assert memory.window_size == 15
        assert memory.message_repo == mock_repo

    def test_initialization_default_window_size(self):
        """Test default window size"""
        # Arrange
        from app.services.memory.sliding_window import SlidingWindowMemory
        from app.repositories.message_repository import MessageRepository

        mock_repo = Mock(spec=MessageRepository)

        # Act
        memory = SlidingWindowMemory(message_repo=mock_repo)

        # Assert
        assert memory.window_size == 10  # Default value

    @pytest.mark.asyncio
    async def test_empty_session_context(self):
        """Test getting context for empty session"""
        # Arrange
        from app.services.memory.sliding_window import SlidingWindowMemory
        from app.repositories.message_repository import MessageRepository

        mock_repo = Mock(spec=MessageRepository)
        memory = SlidingWindowMemory(message_repo=mock_repo, window_size=10)

        mock_repo.get_recent_messages = AsyncMock(return_value=[])

        # Act
        context = await memory.get_context(session_id=1)

        # Assert
        assert context == []
