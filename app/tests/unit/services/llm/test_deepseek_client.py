"""Tests for DeepSeek LLM client"""
import pytest
from unittest.mock import Mock, AsyncMock, patch


class TestDeepSeekClient:
    """Test DeepSeek LLM client implementation"""

    def test_initialization(self):
        """Test client initialization"""
        # Arrange & Act
        from app.services.llm.deepseek_client import DeepSeekClient
        client = DeepSeekClient(
            api_key="test-key",
            model="deepseek-chat",
            max_tokens=1000,
            temperature=0.7
        )

        # Assert
        assert client.api_key == "test-key"
        assert client.model == "deepseek-chat"
        assert client.max_tokens == 1000
        assert client.temperature == 0.7

    def test_initialization_defaults(self):
        """Test client initialization with defaults"""
        # Arrange & Act
        from app.services.llm.deepseek_client import DeepSeekClient
        client = DeepSeekClient(api_key="test-key")

        # Assert
        assert client.model == "deepseek-chat"
        assert client.max_tokens is None
        assert client.temperature is None
        assert client.API_BASE_URL == "https://api.deepseek.com"

    def test_base_url_override(self):
        """Test client with custom base URL"""
        # Arrange & Act
        from app.services.llm.deepseek_client import DeepSeekClient
        client = DeepSeekClient(
            api_key="test-key",
            base_url="https://api.deepseek.com/v1",
        )

        # Assert (openai lib normalizes to a trailing slash)
        assert str(client.client.base_url).rstrip("/") == "https://api.deepseek.com/v1"

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test successful completion generation"""
        # Arrange
        from app.services.llm.deepseek_client import DeepSeekClient
        from app.services.llm.base import LLMMessage

        client = DeepSeekClient(api_key="test-key", model="deepseek-chat")
        messages = [LLMMessage(role="user", content="Hello!")]

        # Mock OpenAI-compatible API
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message.content = "Hello! How can I help you?"
        mock_choice.finish_reason = "stop"
        mock_response.choices = [mock_choice]
        mock_response.model = "deepseek-chat"
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 8
        mock_usage.total_tokens = 18
        mock_response.usage = mock_usage

        with patch.object(client.client.chat.completions, "create",
                          new=AsyncMock(return_value=mock_response)) as mock_create:
            # Act
            response = await client.generate(messages)

        # Assert
        assert response.content == "Hello! How can I help you?"
        assert response.model == "deepseek-chat"
        assert response.usage["total_tokens"] == 18
        mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_stream(self):
        """Test streaming completion generation"""
        # Arrange
        from app.services.llm.deepseek_client import DeepSeekClient
        from app.services.llm.base import LLMMessage

        client = DeepSeekClient(api_key="test-key", model="deepseek-chat")
        messages = [LLMMessage(role="user", content="Hello!")]

        async def chunk_stream():
            for text in ["Hello", " there", "!"]:
                chunk = Mock()
                choice = Mock()
                delta = Mock()
                delta.content = text
                choice.delta = delta
                chunk.choices = [choice]
                yield chunk

        with patch.object(client.client.chat.completions, "create",
                          new=AsyncMock(return_value=chunk_stream())):
            # Act
            chunks = [c async for c in client.generate_stream(messages)]

        # Assert
        assert chunks == ["Hello", " there", "!"]

    @pytest.mark.asyncio
    async def test_generate_error(self):
        """Test error handling on API failure"""
        # Arrange
        from app.services.llm.deepseek_client import DeepSeekClient
        from app.services.llm.base import LLMMessage
        from app.core.exceptions import ExternalServiceError

        client = DeepSeekClient(api_key="test-key", model="deepseek-chat")
        messages = [LLMMessage(role="user", content="Hello!")]

        with patch.object(client.client.chat.completions, "create",
                          new=AsyncMock(side_effect=Exception("API Error"))):
            # Act & Assert
            with pytest.raises(ExternalServiceError):
                await client.generate(messages)

    def test_estimate_tokens(self):
        """Test token estimation"""
        # Arrange
        from app.services.llm.deepseek_client import DeepSeekClient
        client = DeepSeekClient(api_key="test-key")

        # Act
        count = client.estimate_tokens("Hello, world!")

        # Assert
        assert count > 0

    def test_get_max_context_tokens(self):
        """Test max context window"""
        # Arrange
        from app.services.llm.deepseek_client import DeepSeekClient
        client = DeepSeekClient(api_key="test-key", model="deepseek-chat")

        # Act & Assert
        assert client.get_max_context_tokens() == 65536
