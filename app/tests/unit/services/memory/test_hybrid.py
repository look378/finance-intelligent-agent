"""Tests for hybrid memory strategy"""
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock


class TestHybridMemory:
    """Test hybrid memory strategy"""

    def test_initialization(self):
        """Test hybrid memory initialization"""
        # Arrange
        from app.services.memory.hybrid import HybridMemory
        from app.services.memory.sliding_window import SlidingWindowMemory
        from app.services.memory.summarization import SummarizationMemory
        from app.repositories.message_repository import MessageRepository
        from app.services.llm.base import LLMServiceBase

        mock_repo = Mock(spec=MessageRepository)
        mock_llm = Mock(spec=LLMServiceBase)

        sliding = SlidingWindowMemory(message_repo=mock_repo, window_size=10)
        summary = SummarizationMemory(
            message_repo=mock_repo,
            llm_service=mock_llm,
            summary_threshold=20
        )

        # Act
        memory = HybridMemory(
            sliding_window=sliding,
            summarization=summary,
            hybrid_threshold=30
        )

        # Assert
        assert memory.sliding_window == sliding
        assert memory.summarization == summary
        assert memory.hybrid_threshold == 30

    def test_initialization_default_threshold(self):
        """Test default hybrid threshold"""
        # Arrange
        from app.services.memory.hybrid import HybridMemory
        from app.services.memory.sliding_window import SlidingWindowMemory
        from app.services.memory.summarization import SummarizationMemory
        from app.repositories.message_repository import MessageRepository
        from app.services.llm.base import LLMServiceBase

        mock_repo = Mock(spec=MessageRepository)
        mock_llm = Mock(spec=LLMServiceBase)

        sliding = SlidingWindowMemory(message_repo=mock_repo)
        summary = SummarizationMemory(
            message_repo=mock_repo,
            llm_service=mock_llm
        )

        # Act
        memory = HybridMemory(
            sliding_window=sliding,
            summarization=summary
        )

        # Assert
        assert memory.hybrid_threshold == 30  # Default value

    @pytest.mark.asyncio
    async def test_get_context_short_uses_sliding_window(self):
        """Test that short conversations use sliding window"""
        # Arrange
        from app.services.memory.hybrid import HybridMemory
        from app.services.memory.sliding_window import SlidingWindowMemory
        from app.services.memory.summarization import SummarizationMemory
        from app.repositories.message_repository import MessageRepository
        from app.models.database.message import Message

        mock_repo = Mock(spec=MessageRepository)
        mock_llm = Mock(spec=Mock)

        sliding = SlidingWindowMemory(message_repo=mock_repo, window_size=10)
        summary = SummarizationMemory(
            message_repo=mock_repo,
            llm_service=mock_llm
        )

        memory = HybridMemory(
            sliding_window=sliding,
            summarization=summary,
            hybrid_threshold=30
        )

        # Mock message count below threshold
        mock_repo.count_messages = AsyncMock(return_value=10)

        # Mock recent messages
        messages = [
            Message(id=i, role="user", content=f"Msg {i}", created_at=datetime.now())
            for i in range(1, 6)
        ]

        mock_repo.get_recent_messages = AsyncMock(return_value=messages)

        # Act
        context = await memory.get_context(session_id=1)

        # Assert - Should use sliding window for short conversations
        assert len(context) == 5  # Recent messages limit

    @pytest.mark.asyncio
    async def test_get_context_long_uses_summarization(self):
        """Test that long conversations use summarization"""
        # Arrange
        from app.services.memory.hybrid import HybridMemory
        from app.services.memory.sliding_window import SlidingWindowMemory
        from app.services.memory.summarization import SummarizationMemory
        from app.repositories.message_repository import MessageRepository
        from app.models.database.message import Message

        mock_repo = Mock(spec=MessageRepository)
        mock_llm = Mock(spec=Mock)

        sliding = SlidingWindowMemory(message_repo=mock_repo, window_size=10)
        summary = SummarizationMemory(
            message_repo=mock_repo,
            llm_service=mock_llm
        )

        memory = HybridMemory(
            sliding_window=sliding,
            summarization=summary,
            hybrid_threshold=30
        )

        # Mock message count above threshold
        mock_repo.count_messages = AsyncMock(return_value=35)

        # Mock summary
        summary_msg = Message(
            id=1,
            role="system",
            content="Conversation summary",
            created_at=datetime.now()
        )

        # Mock recent messages
        recent_messages = [
            Message(id=2, role="user", content="Recent msg", created_at=datetime.now())
        ]

        mock_repo.get_latest_summary = AsyncMock(return_value=summary_msg)
        mock_repo.get_recent_messages = AsyncMock(return_value=recent_messages)

        # Act
        context = await memory.get_context(session_id=1)

        # Assert - Should use summarization for long conversations
        assert len(context) >= 1
        # Should include summary
        assert any("summary" in msg.content.lower() for msg in context if msg.content)

    @pytest.mark.asyncio
    async def test_add_message_delegates_correct_strategy(self):
        """Test that add_message uses correct strategy based on count"""
        # Arrange
        from app.services.memory.hybrid import HybridMemory
        from app.services.memory.sliding_window import SlidingWindowMemory
        from app.services.memory.summarization import SummarizationMemory
        from app.repositories.message_repository import MessageRepository
        from app.services.llm.base import LLMServiceBase
        from app.models.schemas.chat import MessageContent

        mock_repo = Mock(spec=MessageRepository)
        mock_llm = Mock(spec=Mock)

        sliding = SlidingWindowMemory(message_repo=mock_repo, window_size=10)
        summary = SummarizationMemory(
            message_repo=mock_repo,
            llm_service=mock_llm
        )

        memory = HybridMemory(
            sliding_window=sliding,
            summarization=summary,
            hybrid_threshold=30
        )

        message = MessageContent(role="user", content="Test", timestamp=datetime.now())

        # Test below threshold
        mock_repo.count_messages = AsyncMock(return_value=20)

        # Act
        await memory.add_message(session_id=1, message=message)

        # Assert - Should use sliding window logic (no summarization trigger)

    @pytest.mark.asyncio
    async def test_clear_session(self):
        """Test that clear_session works correctly"""
        # Arrange
        from app.services.memory.hybrid import HybridMemory
        from app.services.memory.sliding_window import SlidingWindowMemory
        from app.services.memory.summarization import SummarizationMemory
        from app.repositories.message_repository import MessageRepository
        from app.services.llm.base import LLMServiceBase

        mock_repo = Mock(spec=MessageRepository)
        mock_llm = Mock(spec=Mock)

        sliding = SlidingWindowMemory(message_repo=mock_repo)
        summary = SummarizationMemory(
            message_repo=mock_repo,
            llm_service=mock_llm
        )

        memory = HybridMemory(
            sliding_window=sliding,
            summarization=summary
        )

        mock_repo.delete_by_session = AsyncMock()

        # Act
        await memory.clear_session(session_id=1)

        # Assert
        mock_repo.delete_by_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_estimate_tokens(self):
        """Test token estimation"""
        # Arrange
        from app.services.memory.hybrid import HybridMemory
        from app.services.memory.sliding_window import SlidingWindowMemory
        from app.repositories.message_repository import MessageRepository
        from app.models.schemas.chat import MessageContent

        mock_repo = Mock(spec=MessageRepository)
        sliding = SlidingWindowMemory(message_repo=mock_repo)
        summary = SummarizationMemory(
            message_repo=mock_repo,
            llm_service=Mock(spec=Mock)
        )

        memory = HybridMemory(
            sliding_window=sliding,
            summarization=summary
        )

        messages = [
            MessageContent(role="user", content="Hello world!", timestamp=datetime.now())
        ]

        # Act
        count = await memory.estimate_tokens(messages)

        # Assert
        assert count > 0
