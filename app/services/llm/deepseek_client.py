"""
DeepSeek LLM client implementation.

Provides integration with DeepSeek models via their OpenAI-compatible API
(base_url: https://api.deepseek.com).  DeepSeek exposes an OpenAI-compatible
endpoint, so this client mirrors the OpenAIClient structure with a
DeepSeek-specific base URL and model defaults.
"""
from typing import AsyncGenerator, Optional, List

from openai import AsyncOpenAI

from app.services.llm.base import LLMServiceBase, LLMMessage, LLMResponse
from app.services.llm.token_counter import TokenCounter
from app.core.exceptions import ExternalServiceError


class DeepSeekClient(LLMServiceBase):
    """
    DeepSeek LLM service implementation (OpenAI-compatible API).

    Supports deepseek-chat and deepseek-reasoner models.
    """

    # Available DeepSeek models
    MODELS = [
        "deepseek-chat",
        "deepseek-reasoner",
    ]

    # DeepSeek OpenAI-compatible API base URL
    API_BASE_URL = "https://api.deepseek.com"

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        base_url: Optional[str] = None,
    ) -> None:
        """
        Initialize DeepSeek client.

        Args:
            api_key: DeepSeek API key
            model: Model name (default: deepseek-chat)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            base_url: Optional override of the API base URL
        """
        super().__init__(api_key, model, max_tokens, temperature)

        # Initialize async OpenAI-compatible client pointed at DeepSeek
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or self.API_BASE_URL,
        )

    async def generate(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a completion from DeepSeek.

        Args:
            messages: List of conversation messages
            max_tokens: Override max tokens
            temperature: Override temperature
            **kwargs: Additional parameters (passed to the OpenAI-compatible API)

        Returns:
            LLMResponse: Generated response

        Raises:
            ExternalServiceError: If API call fails
        """
        try:
            params = {
                "model": self.model,
                "messages": self._format_messages(messages),
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature,
            }
            params.update(kwargs)
            params = {k: v for k, v in params.items() if v is not None}

            response = await self.client.chat.completions.create(**params)

            choice = response.choices[0]
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return LLMResponse(
                content=choice.message.content,
                model=response.model,
                finish_reason=choice.finish_reason,
                usage=usage,
            )

        except Exception as e:
            raise ExternalServiceError(
                service="DeepSeek",
                message=f"Failed to generate completion: {str(e)}",
            ) from e

    async def generate_stream(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """
        Generate a streaming completion from DeepSeek.

        Args:
            messages: List of conversation messages
            max_tokens: Override max tokens
            temperature: Override temperature
            **kwargs: Additional parameters

        Yields:
            str: Content chunks

        Raises:
            ExternalServiceError: If API call fails
        """
        try:
            params = {
                "model": self.model,
                "messages": self._format_messages(messages),
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature,
                "stream": True,
            }
            params.update(kwargs)
            params = {k: v for k, v in params.items() if v is not None}

            stream = await self.client.chat.completions.create(**params)

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise ExternalServiceError(
                service="DeepSeek",
                message=f"Failed to generate streaming completion: {str(e)}",
            ) from e

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using character-based estimation.

        Args:
            text: Text to estimate

        Returns:
            int: Estimated token count
        """
        return TokenCounter.estimate(text)

    async def count_tokens(self, messages: List[LLMMessage]) -> int:
        """
        Count actual tokens in messages.

        Args:
            messages: List of messages

        Returns:
            int: Total token count
        """
        return TokenCounter.count_messages(messages, self.model)

    def get_max_context_tokens(self) -> int:
        """
        Get the maximum context window for the current model.

        Returns:
            int: Maximum tokens in context window
        """
        context_windows = {
            "deepseek-chat": 65536,
            "deepseek-reasoner": 65536,
        }
        return context_windows.get(self.model, 65536)
