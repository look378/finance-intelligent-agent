"""Tests for token counter utilities"""
import pytest


class TestTokenCounter:
    """Test token counting utilities"""

    def test_count_tokens_empty_string(self):
        """Test counting tokens in empty string"""
        # Arrange
        from app.services.llm.token_counter import TokenCounter

        # Act
        count = TokenCounter.count("")

        # Assert
        assert count == 0

    def test_count_tokens_simple_text(self):
        """Test counting tokens in simple text"""
        # Arrange
        from app.services.llm.token_counter import TokenCounter
        text = "Hello, world!"

        # Act
        count = TokenCounter.count(text)

        # Assert
        # Rough estimate: ~4 characters per token
        # "Hello, world!" = 13 chars ≈ 3-4 tokens
        assert count > 0
        assert count < 10

    def test_count_tokens_long_text(self):
        """Test counting tokens in longer text"""
        # Arrange
        from app.services.llm.token_counter import TokenCounter
        text = "This is a longer piece of text that should have more tokens. " * 10

        # Act
        count = TokenCounter.count(text)

        # Assert
        assert count > 50

    def test_count_tokens_with_special_chars(self):
        """Test counting tokens with special characters"""
        # Arrange
        from app.services.llm.token_counter import TokenCounter
        text = "Hello! @#$%^&*()_+-=[]{}|;':\",./<>?"

        # Act
        count = TokenCounter.count(text)

        # Assert
        assert count > 0

    def test_count_tokens_code(self):
        """Test counting tokens in code"""
        # Arrange
        from app.services.llm.token_counter import TokenCounter
        code = """
        def hello_world():
            print("Hello, world!")
            return True
        """

        # Act
        count = TokenCounter.count(code)

        # Assert
        assert count > 0

    def test_estimate_tokens_simple(self):
        """Test simple token estimation"""
        # Arrange
        from app.services.llm.token_counter import TokenCounter
        text = "Hello world"

        # Act
        estimate = TokenCounter.estimate(text)

        # Assert
        # Should be close to character count / 4
        expected = len(text) / 4
        assert abs(estimate - expected) < 2

    def test_estimate_vs_count(self):
        """Test that estimate is close to actual count"""
        # Arrange
        from app.services.llm.token_counter import TokenCounter
        text = "This is a sample text for testing token counting and estimation."

        # Act
        count = TokenCounter.count(text)
        estimate = TokenCounter.estimate(text)

        # Assert
        # Estimate should be within 50% of actual count
        ratio = estimate / count if count > 0 else 1
        assert 0.5 <= ratio <= 1.5

    def test_count_messages_empty(self):
        """Test counting tokens in empty message list"""
        # Arrange
        from app.services.llm.token_counter import TokenCounter
        from app.services.llm.base import LLMMessage
        messages = []

        # Act
        count = TokenCounter.count_messages(messages)

        # Assert
        assert count == 0

    def test_count_messages_single(self):
        """Test counting tokens in single message"""
        # Arrange
        from app.services.llm.token_counter import TokenCounter
        from app.services.llm.base import LLMMessage
        messages = [
            LLMMessage(role="user", content="Hello, world!")
        ]

        # Act
        count = TokenCounter.count_messages(messages)

        # Assert
        assert count > 0

    def test_count_messages_multiple(self):
        """Test counting tokens in multiple messages"""
        # Arrange
        from app.services.llm.token_counter import TokenCounter
        from app.services.llm.base import LLMMessage
        messages = [
            LLMMessage(role="system", content="You are a helpful assistant."),
            LLMMessage(role="user", content="Hello! How are you?"),
            LLMMessage(role="assistant", content="I'm doing well, thank you!"),
        ]

        # Act
        count = TokenCounter.count_messages(messages)

        # Assert
        assert count > 10

    def test_count_tokens_multilingual(self):
        """Test counting tokens in multilingual text"""
        # Arrange
        from app.services.llm.token_counter import TokenCounter
        text = "Hello 你好 مرحبا 안녕하세요"

        # Act
        count = TokenCounter.count(text)

        # Assert
        assert count > 0

    def test_count_tokens_numbers(self):
        """Test counting tokens with numbers"""
        # Arrange
        from app.services.llm.token_counter import TokenCounter
        text = "The numbers are: 123, 4567, and 89.012"

        # Act
        count = TokenCounter.count(text)

        # Assert
        assert count > 0
