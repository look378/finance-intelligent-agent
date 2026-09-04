"""Tests for memory factory"""
import pytest
from unittest.mock import Mock


class TestMemoryFactory:
    """Test memory factory for creating memory strategies"""

    def test_create_sliding_window(self):
        """Test creating sliding window memory"""
        # Arrange
        from app.services.memory.factory import MemoryFactory
        from app.repositories.message_repository import MessageRepository

        mock_repo = Mock(spec=MessageRepository)

        # Act
        memory = MemoryFactory.create(
            memory_type="sliding_window",
            message_repo=mock_repo,
            window_size=15
        )

        # Assert
        from app.services.memory.sliding_window import SlidingWindowMemory
        assert isinstance(memory, SlidingWindowMemory)
        assert memory.window_size == 15

    def test_create_summarization(self):
        """Test creating summarization memory"""
        # Arrange
        from app.services.memory.factory import MemoryFactory
        from app.repositories.message_repository import MessageRepository
        from app.services.llm.base import LLMServiceBase

        mock_repo = Mock(spec=MessageRepository)
        mock_llm = Mock(spec=LLMServiceBase)

        # Act
        memory = MemoryFactory.create(
            memory_type="summarization",
            message_repo=mock_repo,
            llm_service=mock_llm,
            summary_threshold=25
        )

        # Assert
        from app.services.memory.summarization import SummarizationMemory
        assert isinstance(memory, SummarizationMemory)
        assert memory.summary_threshold == 25

    def test_create_hybrid(self):
        """Test creating hybrid memory"""
        # Arrange
        from app.services.memory.factory import MemoryFactory
        from app.repositories.message_repository import MessageRepository
        from app.services.llm.base import LLMServiceBase

        mock_repo = Mock(spec=MessageRepository)
        mock_llm = Mock(spec=LLMServiceBase)

        # Act
        memory = MemoryFactory.create(
            memory_type="hybrid",
            message_repo=mock_repo,
            llm_service=mock_llm,
            window_size=10,
            summary_threshold=20,
            hybrid_threshold=35
        )

        # Assert
        from app.services.memory.hybrid import HybridMemory
        assert isinstance(memory, HybridMemory)
        assert memory.hybrid_threshold == 35

    def test_create_invalid_memory_type(self):
        """Test creating invalid memory type raises error"""
        # Arrange
        from app.services.memory.factory import MemoryFactory
        from app.repositories.message_repository import MessageRepository
        from app.core.exceptions import ValidationError

        mock_repo = Mock(spec=MessageRepository)

        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            MemoryFactory.create(
                memory_type="invalid_type",
                message_repo=mock_repo
            )
        assert "memory" in str(exc_info.value).lower()

    def test_create_sliding_window_defaults(self):
        """Test creating sliding window with default parameters"""
        # Arrange
        from app.services.memory.factory import MemoryFactory
        from app.repositories.message_repository import MessageRepository

        mock_repo = Mock(spec=MessageRepository)

        # Act
        memory = MemoryFactory.create(
            memory_type="sliding_window",
            message_repo=mock_repo
        )

        # Assert
        from app.services.memory.sliding_window import SlidingWindowMemory
        assert isinstance(memory, SlidingWindowMemory)
        assert memory.window_size == 10  # Default

    def test_create_summarization_with_defaults(self):
        """Test creating summarization with default parameters"""
        # Arrange
        from app.services.memory.factory import MemoryFactory
        from app.repositories.message_repository import MessageRepository
        from app.services.llm.base import LLMServiceBase

        mock_repo = Mock(spec=MessageRepository)
        mock_llm = Mock(spec=LLMServiceBase)

        # Act
        memory = MemoryFactory.create(
            memory_type="summarization",
            message_repo=mock_repo,
            llm_service=mock_llm
        )

        # Assert
        from app.services.memory.summarization import SummarizationMemory
        assert isinstance(memory, SummarizationMemory)
        assert memory.summary_threshold == 20  # Default
        assert memory.summary_interval == 10  # Default

    def test_create_hybrid_with_defaults(self):
        """Test creating hybrid with default parameters"""
        # Arrange
        from app.services.memory.factory import MemoryFactory
        from app.repositories.message_repository import MessageRepository
        from app.services.llm.base import LLMServiceBase

        mock_repo = Mock(spec=MessageRepository)
        mock_llm = Mock(spec=LLMServiceBase)

        # Act
        memory = MemoryFactory.create(
            memory_type="hybrid",
            message_repo=mock_repo,
            llm_service=mock_llm
        )

        # Assert
        from app.services.memory.hybrid import HybridMemory
        assert isinstance(memory, HybridMemory)
        assert memory.hybrid_threshold == 30  # Default
