"""
Token counting utilities for LLM interactions.

Provides token estimation and counting for different LLM providers.
"""
import re
from typing import List

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


class TokenCounter:
    """
    Token counting utilities for LLM token estimation.

    Uses tiktoken library when available for accurate counting,
    falls back to estimation when not available.
    """

    # Approximate characters per token for different models
    CHARS_PER_TOKEN = {
        "gpt-3.5-turbo": 4,
        "gpt-4": 4,
        "gpt-4-turbo": 4,
        "text-embedding-3-small": 4,
        "text-embedding-3-large": 4,
        "claude-3": 4,
        "default": 4,
    }

    @staticmethod
    def count(text: str, model: str = "gpt-4") -> int:
        """
        Count the number of tokens in text.

        Uses tiktoken for accurate counting when available,
        otherwise estimates based on character count.

        Args:
            text: Text to count tokens in
            model: Model name for tokenizer selection

        Returns:
            int: Number of tokens
        """
        if not text:
            return 0

        if TIKTOKEN_AVAILABLE:
            try:
                # Get encoding for the model
                try:
                    encoding = tiktoken.encoding_for_model(model)
                except KeyError:
                    # Fall back to cl100k_base (GPT-4) encoding
                    encoding = tiktoken.get_encoding("cl100k_base")

                # Count tokens
                tokens = encoding.encode(text)
                return len(tokens)
            except Exception:
                # Fall back to estimation on error
                pass

        # Estimate: roughly 4 characters per token
        return TokenCounter.estimate(text)

    @staticmethod
    def estimate(text: str) -> int:
        """
        Estimate token count based on character count.

        Args:
            text: Text to estimate tokens for

        Returns:
            int: Estimated token count
        """
        if not text:
            return 0

        # Rough estimate: ~4 characters per token
        # Adjusted for spaces and punctuation
        return len(text) // 4

    @staticmethod
    def count_messages(messages: list, model: str = "gpt-4") -> int:
        """
        Count total tokens in a list of messages.

        Args:
            messages: List of LLMMessage objects
            model: Model name for tokenizer selection

        Returns:
            int: Total token count across all messages
        """
        if not messages:
            return 0

        # Count tokens in each message
        total = 0
        for message in messages:
            # Add tokens for role and content
            # Each message has overhead (~4 tokens for format)
            total += 4  # Message format overhead
            total += TokenCounter.count(message.content, model)

        # Add reply tokens (for the assistant's response)
        total += 3  # Reply priming tokens

        return total

    @staticmethod
    def estimate_messages(messages: list) -> int:
        """
        Estimate total tokens in messages without tiktoken.

        Args:
            messages: List of LLMMessage objects

        Returns:
            int: Estimated token count
        """
        if not messages:
            return 0

        total = 0
        for message in messages:
            # Message format overhead (~4 tokens)
            total += 4
            # Content tokens
            total += TokenCounter.estimate(message.content)

        # Reply tokens
        total += 3

        return total

    @staticmethod
    def calculate_max_tokens(
        messages: list,
        model: str = "gpt-4",
        max_context_tokens: int = 128000,
        reserve_tokens: int = 1000,
    ) -> int:
        """
        Calculate appropriate max_tokens for completion.

        Args:
            messages: List of messages in conversation
            model: Model name
            max_context_tokens: Maximum context window for model
            reserve_tokens: Tokens to reserve for response overhead

        Returns:
            int: Recommended max_tokens value
        """
        # Count tokens in messages
        message_tokens = TokenCounter.count_messages(messages, model)

        # Calculate remaining tokens
        remaining = max_context_tokens - message_tokens - reserve_tokens

        # Ensure at least some tokens for response
        return max(100, min(remaining, max_context_tokens // 2))

    @staticmethod
    def truncate_messages_by_tokens(
        messages: list,
        max_tokens: int,
        model: str = "gpt-4",
    ) -> list:
        """
        Truncate messages to fit within token limit.

        Keeps most recent messages (end of list) and removes older ones.

        Args:
            messages: List of messages
            max_tokens: Maximum tokens allowed
            model: Model name for tokenizer

        Returns:
            list: Truncated list of messages
        """
        if not messages:
            return []

        # Start with most recent messages
        truncated = []
        current_tokens = 0

        # Add messages from newest to oldest
        for message in reversed(messages):
            message_tokens = TokenCounter.count(message.content, model) + 4  # +4 for format

            if current_tokens + message_tokens > max_tokens:
                break

            truncated.insert(0, message)
            current_tokens += message_tokens

        return truncated
