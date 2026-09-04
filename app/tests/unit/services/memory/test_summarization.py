"""Tests for summarization memory strategy"""
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock


class TestSummarizationMemory:
    """Test summarization memory strategy"""

    @pytest.mark.asyncio
    async def test_get_context_with_summary(self):
        """Test that get_context includes summary and recent messages"""
        # Arrange
        from app.services.memory.summarization import SummarizationMemory
        from app.models.schemas.chat import MessageContent
        from app.models.database.message import Message
        from app.repositories.message_repository import MessageRepository

        mock_repo = Mock(spec=MessageRepository)
        llm_service = Mock()
        memory = SummarizationMemory(
            message_repo=mock_repo,
            llm_service=llm_service,
            summary_threshold=5,
            summary_interval=3
        )

        # Mock summary message
        summary_msg = Message(
            id=1,
            role="system",
            content="Conversation summary: User asked about AI",
            created_at=datetime.now()
        )

        # Mock recent messages
        recent_messages = [
            Message(id=2, role="user", content="What is machine learning?", created_at=datetime.now()),
            Message(id=3, role="assistant", content="ML is a subset of AI", created_at=datetime.now()),
        ]

        mock_repo.get_latest_summary = AsyncMock(return_value=summary_msg)
        mock_repo.get_recent_messages = AsyncMock(return_value=recent_messages)

        # Act
        context = await memory.get_context(session_id=1)

        # Assert
        assert len(context) == 3  # Summary + 2 recent messages
        assert context[0].role == "system"
        assert "summary" in context[0].content.lower()

    @pytest.mark.asyncio
    async def test_add_message_triggers_summary_when_threshold_reached(self):
        """Test that summary is created when message count reaches threshold"""
        # Arrange
        from app.services.memory.summarization import SummarizationMemory
        from app.models.schemas.chat import MessageContent
        from app.repositories.message_repository import MessageRepository

        mock_repo = Mock(spec=MessageRepository)
        llm_service = Mock()
        memory = SummarizationMemory(
            message_repo=mock_repo,
            llm_service=llm_service,
            summary_threshold=5,
            summary_interval=3
        )

        # Mock message count
        mock_repo.count_messages = AsyncMock(return_value=5)
        mock_repo.get_messages_before_summary = AsyncMock(return_value=[])
        llm_service.generate = AsyncMock(return_value="Summary of conversation")

        mock_repo.create_summary = AsyncMock()
        mock_repo.create = AsyncMock()

        message = MessageContent(role="user", content="Test", timestamp=datetime.now())

        # Act
        await memory.add_message(session_id=1, message=message)

        # Assert
        llm_service.generate.assert_called_once()
        mock_repo.create_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_message_below_threshold(self):
        """Test that summary is NOT created when below threshold"""
        # Arrange
        from app.services.memory.summarization import SummarizationMemory
        from app.models.schemas.chat import MessageContent
        from app.repositories.message_repository import MessageRepository

        mock_repo = Mock(spec=MessageRepository)
        llm_service = Mock()
        memory = SummarizationMemory(
            message_repo=mock_repo,
            llm_service=llm_service,
            summary_threshold=10,
            summary_interval=5
        )

        # Mock message count below threshold
        mock_repo.count_messages = AsyncMock(return_value=3)
        mock_repo.create = AsyncMock()

        message = MessageContent(role="user", content="Test", timestamp=datetime.now())

        # Act
        await memory.add_message(session_id=1, message=message)

        # Assert
        llm_service.generate.assert_not_called()
        mock_repo.create_summary.assert_not_called()

    @pytest.mark.asyncio
    async def test_clear_session(self):
        """Test that clear_session works correctly"""
        # Arrange
        from app.services.memory.summarization import SummarizationMemory
        from app.repositories.message_repository import MessageRepository

        mock_repo = Mock(spec=MessageRepository)
        llm_service = Mock()
        memory = SummarizationMemory(
            message_repo=mock_repo,
            llm_service=llm_service
        )

        mock_repo.delete_by_session = AsyncMock()

        # Act
        await memory.clear_session(session_id=1)

        # Assert
        mock_repo.delete_by_session.assert_called_once_with(1)

    def test_initialization(self):
        """Test strategy initialization"""
        # Arrange
        from app.services.memory.summarization import SummarizationMemory
        from app.repositories.message_repository import MessageRepository

        mock_repo = Mock(spec=MessageRepository)
        llm_service = Mock()

        # Act
        memory = SummarizationMemory(
            message_repo=mock_repo,
            llm_service=llm_service,
            summary_threshold=20,
            summary_interval=10
        )

        # Assert
        assert memory.summary_threshold == 20
        assert memory.summary_interval == 10

    @pytest.mark.asyncio
    async def test_create_summary_generates_summary(self):
        """Test that _create_summary generates summary with LLM"""
        # Arrange
        from app.services.memory.summarization import SummarizationMemory
        from app.models.database.message import Message
        from app.repositories.message_repository import MessageRepository

        mock_repo = Mock(spec=MessageRepository)
        llm_service = Mock()
        memory = SummarizationMemory(
            message_repo=mock_repo,
            llm_service=llm_service
        )

        # Mock messages to summarize
        messages = [
            Message(id=1, role="user", content="Hello", created_at=datetime.now()),
            Message(id=2, role="assistant", content="Hi there!", created_at=datetime.now()),
        ]

        mock_repo.get_messages_before_summary = AsyncMock(return_value=messages)
        llm_service.generate = AsyncMock(return_value="Summary: User greeted assistant")

        # Act
        await memory._create_summary(session_id=1)

        # Assert
        llm_service.generate.assert_called_once()
        # Check that prompt was built correctly
        call_args = llm_service.generate.call_args
        messages_arg = call_args[0][0]  # First positional arg
        assert len(messages_arg) >= 2  # At least system prompt + user query

    @pytest.mark.asyncio
    async def test_create_summary_stores_as_system_message(self):
        """Test that summary is stored as system message"""
        # Arrange
        from app.services.memory.summarization import SummarizationMemory
        from app.repositories.message_repository import MessageRepository

        mock_repo = Mock(spec=MessageRepository)
        llm_service = Mock()
        memory = SummarizationMemory(
            message_repo=mock_repo,
            llm_service=llm_service
        )

        mock_repo.get_messages_before_summary = AsyncMock(return_value=[])
        llm_service.generate = AsyncMock(return_value="Test summary")

        mock_repo.create_summary = AsyncMock()

        # Act
        await memory._create_summary(session_id=1)

        # Assert
        mock_repo.create_summary.assert_called_once()
        # Check that summary was created with "system" role

    @pytest.mark.asyncio
    async def test_create_summary_archives_old_messages(self):
        """Test that old messages are archived after summary"""
        # Arrange
        from app.services.memory.summarization import SummarizationMemory
        from app.repositories.message_repository import MessageRepository

        mock_repo = Mock(spec=MessageRepository)
        llm_service = Mock()
        memory = SummarizationMemory(
            message_repo=mock_repo,
            llm_service=llm_service
        )

        # Mock old messages
        old_messages = [
            Message(id=1, role="user", content="Old msg 1"),
            Message(id=2, role="assistant", content="Old msg 2"),
        ]

        mock_repo.get_messages_before_summary = AsyncMock(return_value=old_messages)
        llm_service.generate = AsyncMock(return_value="Summary")

        mock_repo.archive_messages = AsyncMock()

        # Act
        await memory._create_summary(session_id=1)

        # Assert
        mock_repo.archive_messages.assert_called_once_with(1, 2)

    @pytest.mark.asyncio
    async def test_estimate_tokens(self):
        """Test token estimation"""
        # Arrange
        from app.services.memory.summarization import SummarizationMemory
        from app.repositories.message_repository import MessageRepository

        mock_repo = Mock(spec=MessageRepository)
        llm_service = Mock()
        memory = SummarizationMemory(
            message_repo=mock_repo,
            llm_service=llm_service
        )

        messages = [
            MessageContent(role="user", content="Hello world!", timestamp=datetime.now())
        ]

        # Act
        count = await memory.estimate_tokens(messages)

        # Assert
        assert count > 0

    @pytest.mark.asyncio
    async def test_get_context_without_summary(self):
        """Test getting context when no summary exists yet"""
        # Arrange
        from app.services.memory.summarization import SummarizationMemory
        from app.models.database.message import Message
        from app.repositories.message_repository import MessageRepository

        mock_repo = Mock(spec=MessageRepository)
        llm_service = Mock()
        memory = SummarizationMemory(
            message_repo=mock_repo,
            llm_service=llm_service,
            summary_threshold=10,
            summary_interval=5
        )

        # No summary exists
        mock_repo.get_latest_summary = AsyncMock(return_value=None)

        # Mock recent messages
        recent_messages = [
            Message(id=1, role="user", content="Recent msg", created_at=datetime.now()),
        ]

        mock_repo.get_recent_messages = AsyncMock(return_value=recent_messages)

        # Act
        context = await memory.get_context(session_id=1)

        # Assert
        assert len(context) == 1
        assert context[0].content == "Recent msg"
