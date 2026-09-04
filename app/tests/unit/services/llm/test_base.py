"""Tests for LLM base interface and service"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List


class TestLLMMessage:
    """Test LLM message data structure"""

    def test_create_user_message(self):
        """Test creating a user message"""
        # Arrange & Act
        from app.services.llm.base import LLMMessage
        message = LLMMessage(role="user", content="Hello, world!")

        # Assert
        assert message.role == "user"
        assert message.content == "Hello, world!"

    def test_create_system_message(self):
        """Test creating a system message"""
        # Arrange & Act
        from app.services.llm.base import LLMMessage
        message = LLMMessage(role="system", content="You are a helpful assistant.")

        # Assert
        assert message.role == "system"
        assert message.content == "You are a helpful assistant."

    def test_create_assistant_message(self):
        """Test creating an assistant message"""
        # Arrange & Act
        from app.services.llm.base import LLMMessage
        message = LLMMessage(role="assistant", content="Hi there!")

        # Assert
        assert message.role == "assistant"
        assert message.content == "Hi there!"

    def test_message_to_dict(self):
        """Test converting message to dictionary"""
        # Arrange
        from app.services.llm.base import LLMMessage
        message = LLMMessage(role="user", content="Test")

        # Act
        msg_dict = message.to_dict()

        # Assert
        assert msg_dict == {"role": "user", "content": "Test"}

    def test_message_from_dict(self):
        """Test creating message from dictionary"""
        # Arrange
        from app.services.llm.base import LLMMessage
        msg_dict = {"role": "user", "content": "Test"}

        # Act
        message = LLMMessage.from_dict(msg_dict)

        # Assert
        assert message.role == "user"
        assert message.content == "Test"


class TestLLMResponse:
    """Test LLM response data structure"""

    def test_create_response(self):
        """Test creating an LLM response"""
        # Arrange & Act
        from app.services.llm.base import LLMResponse
        response = LLMResponse(
            content="Hello!",
            model="gpt-4",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )

        # Assert
        assert response.content == "Hello!"
        assert response.model == "gpt-4"
        assert response.usage["total_tokens"] == 15

    def test_response_with_finish_reason(self):
        """Test response with finish reason"""
        # Arrange & Act
        from app.services.llm.base import LLMResponse
        response = LLMResponse(
            content="Test",
            model="gpt-4",
            finish_reason="stop"
        )

        # Assert
        assert response.finish_reason == "stop"


class TestLLMServiceBase:
    """Test LLM service base class"""

    @pytest.mark.asyncio
    async def test_generate_not_implemented(self):
        """Test that generate raises NotImplementedError in base class"""
        # Arrange
        from app.services.llm.base import LLMServiceBase
        from app.services.llm.base import LLMMessage

        service = LLMServiceBase()
        messages = [LLMMessage(role="user", content="Test")]

        # Act & Assert
        with pytest.raises(NotImplementedError):
            await service.generate(messages=messages)

    @pytest.mark.asyncio
    async def test_generate_stream_not_implemented(self):
        """Test that generate_stream raises NotImplementedError in base class"""
        # Arrange
        from app.services.llm.base import LLMServiceBase
        from app.services.llm.base import LLMMessage

        service = LLMServiceBase()
        messages = [LLMMessage(role="user", content="Test")]

        # Act & Assert
        with pytest.raises(NotImplementedError):
            async for _ in service.generate_stream(messages=messages):
                pass

    def test_estimate_tokens_not_implemented(self):
        """Test that estimate_tokens raises NotImplementedError in base class"""
        # Arrange
        from app.services.llm.base import LLMServiceBase

        service = LLMServiceBase()
        text = "Sample text"

        # Act & Assert
        with pytest.raises(NotImplementedError):
            service.estimate_tokens(text)

    @pytest.mark.asyncio
    async def test_count_tokens_not_implemented(self):
        """Test that count_tokens raises NotImplementedError in base class"""
        # Arrange
        from app.services.llm.base import LLMServiceBase
        from app.services.llm.base import LLMMessage

        service = LLMServiceBase()
        messages = [LLMMessage(role="user", content="Test")]

        # Act & Assert
        with pytest.raises(NotImplementedError):
            await service.count_tokens(messages)

    def test_validate_max_tokens(self):
        """Test max_tokens validation"""
        # Arrange
        from app.services.llm.base import LLMServiceBase

        service = LLMServiceBase()

        # Act & Assert - Valid max_tokens
        assert service._validate_max_tokens(100) == 100
        assert service._validate_max_tokens(1) == 1
        assert service._validate_max_tokens(None) is None

    def test_validate_max_tokens_invalid(self):
        """Test max_tokens validation with invalid values"""
        # Arrange
        from app.services.llm.base import LLMServiceBase

        service = LLMServiceBase()

        # Act & Assert - Invalid max_tokens
        with pytest.raises(ValueError):
            service._validate_max_tokens(0)

        with pytest.raises(ValueError):
            service._validate_max_tokens(-1)

    def test_validate_temperature(self):
        """Test temperature validation"""
        # Arrange
        from app.services.llm.base import LLMServiceBase

        service = LLMServiceBase()

        # Act & Assert - Valid temperatures
        assert service._validate_temperature(0.5) == 0.5
        assert service._validate_temperature(0.0) == 0.0
        assert service._validate_temperature(1.0) == 1.0
        assert service._validate_temperature(None) is None

    def test_validate_temperature_invalid(self):
        """Test temperature validation with invalid values"""
        # Arrange
        from app.services.llm.base import LLMServiceBase

        service = LLMServiceBase()

        # Act & Assert - Invalid temperatures
        with pytest.raises(ValueError):
            service._validate_temperature(-0.1)

        with pytest.raises(ValueError):
            service._validate_temperature(1.1)
