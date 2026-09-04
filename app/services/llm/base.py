"""
Base LLM service interface and data structures.

Provides abstract interfaces for LLM providers to implement.
"""
from abc import ABC, abstractmethod
from typing import Optional, AsyncGenerator, List, Dict, Any
from dataclasses import dataclass


@dataclass
class LLMMessage:
    """
    Represents a message in a conversation with an LLM.

    Attributes:
        role: Message role ("user", "assistant", or "system")
        content: Message content
    """
    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        """
        Convert message to dictionary format.

        Returns:
            dict: Message as dictionary with role and content
        """
        return {
            "role": self.role,
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "LLMMessage":
        """
        Create message from dictionary.

        Args:
            data: Dictionary with role and content

        Returns:
            LLMMessage: Message instance
        """
        return cls(role=data["role"], content=data["content"])


@dataclass
class LLMResponse:
    """
    Represents a response from an LLM.

    Attributes:
        content: Generated text content
        model: Model name/identifier used
        finish_reason: Reason the generation finished
        usage: Token usage information
    """
    content: str
    model: str
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None


class LLMServiceBase(ABC):
    """
    Abstract base class for LLM service implementations.

    All LLM provider implementations (OpenAI, Anthropic, etc.) must
    inherit from this class and implement the required methods.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """
        Initialize the LLM service.

        Args:
            api_key: API key for the LLM provider
            model: Model name/identifier
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0 to 1.0)
        """
        self.api_key = api_key
        self.model = model
        self.max_tokens = self._validate_max_tokens(max_tokens)
        self.temperature = self._validate_temperature(temperature)

    @abstractmethod
    async def generate(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a completion from the LLM.

        Args:
            messages: List of conversation messages
            max_tokens: Override max tokens for this request
            temperature: Override temperature for this request
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse: The generated response

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("generate() must be implemented by subclass")

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming completion from the LLM.

        Args:
            messages: List of conversation messages
            max_tokens: Override max tokens for this request
            temperature: Override temperature for this request
            **kwargs: Additional provider-specific parameters

        Yields:
            str: Chunks of generated content

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("generate_stream() must be implemented by subclass")

    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in text.

        Args:
            text: Text to estimate tokens for

        Returns:
            int: Estimated token count

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("estimate_tokens() must be implemented by subclass")

    @abstractmethod
    async def count_tokens(self, messages: List[LLMMessage]) -> int:
        """
        Count the actual number of tokens in messages.

        Args:
            messages: List of messages to count tokens for

        Returns:
            int: Total token count

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("count_tokens() must be implemented by subclass")

    def _validate_max_tokens(self, max_tokens: Optional[int]) -> Optional[int]:
        """
        Validate max_tokens parameter.

        Args:
            max_tokens: Max tokens value to validate

        Returns:
            Optional[int]: Validated max_tokens or None

        Raises:
            ValueError: If max_tokens is invalid
        """
        if max_tokens is None:
            return None

        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")

        return max_tokens

    def _validate_temperature(self, temperature: Optional[float]) -> Optional[float]:
        """
        Validate temperature parameter.

        Args:
            temperature: Temperature value to validate

        Returns:
            Optional[float]: Validated temperature or None

        Raises:
            ValueError: If temperature is invalid
        """
        if temperature is None:
            return None

        if not (0.0 <= temperature <= 1.0):
            raise ValueError(f"temperature must be between 0.0 and 1.0, got {temperature}")

        return temperature

    def _format_messages(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        """
        Format messages for API request.

        Args:
            messages: List of LLM messages

        Returns:
            List[dict]: Formatted messages
        """
        return [msg.to_dict() for msg in messages]
