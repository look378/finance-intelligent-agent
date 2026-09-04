"""Tests for memory management base interface"""
import pytest
from unittest.mock import Mock
from datetime import datetime


class TestMemoryStrategy:
    """Test Memory strategy base class"""

    async def test_base_class_not_implemented_get_context(self):
        """Test that get_context raises NotImplementedError"""
        from app.services.memory.base import MemoryStrategy

        strategy = MemoryStrategy(message_repo=Mock())

        with pytest.raises(NotImplementedError):
            await strategy.get_context(session_id=1)

    async def test_base_class_not_implemented_add_message(self):
        """Test that add_message raises NotImplementedError"""
        from app.services.memory.base import MemoryStrategy
        from app.models.schemas.chat import MessageContent

        strategy = MemoryStrategy(message_repo=Mock())
        message = MessageContent(
            role="user",
            content="Hello",
            timestamp=datetime.now()
        )

        with pytest.raises(NotImplementedError):
            await strategy.add_message(session_id=1, message=message)

    async def test_base_class_not_implemented_clear_session(self):
        """Test that clear_session raises NotImplementedError"""
        from app.services.memory.base import MemoryStrategy

        strategy = MemoryStrategy(message_repo=Mock())

        with pytest.raises(NotImplementedError):
            await strategy.clear_session(session_id=1)

    def test_base_class_initialization(self):
        """Test base class initialization with message repo"""
        # Arrange
        from app.services.memory.base import MemoryStrategy
        from app.repositories.message_repository import MessageRepository

        mock_repo = Mock(spec=MessageRepository)

        # Act
        strategy = MemoryStrategy(message_repo=mock_repo)

        # Assert
        assert strategy.message_repo == mock_repo

    def test_estimate_tokens_not_implemented(self):
        """Test that estimate_tokens raises NotImplementedError"""
        # Arrange
        from app.services.memory.base import MemoryStrategy
        from app.models.schemas.chat import MessageContent

        strategy = MemoryStrategy(message_repo=Mock())
        messages = [
            MessageContent(role="user", content="Hello", timestamp=datetime.now())
        ]

        # Act & Assert
        with pytest.raises(NotImplementedError):
            strategy.estimate_tokens(messages)


class TestMemoryContent:
    """Test memory content data structures"""

    def test_create_memory_content(self):
        """Test creating memory content"""
        # Arrange
        from app.services.memory.base import MemoryContent

        # Act
        content = MemoryContent(
            messages=[],
            summary="Conversation summary",
            metadata={"token_count": 100}
        )

        # Assert
        assert content.messages == []
        assert content.summary == "Conversation summary"
        assert content.metadata["token_count"] == 100

    def test_create_memory_content_defaults(self):
        """Test creating memory content with defaults"""
        # Arrange
        from app.services.memory.base import MemoryContent

        # Act
        content = MemoryContent(
            messages=[],
        )

        # Assert
        assert content.messages == []
        assert content.summary is None
        assert content.metadata is None
