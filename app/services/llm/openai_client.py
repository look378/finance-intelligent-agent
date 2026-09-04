"""
OpenAI LLM client implementation.

Provides integration with OpenAI's GPT models.
"""
from typing import AsyncGenerator, Optional, List

from openai import AsyncOpenAI

from app.services.llm.base import LLMServiceBase, LLMMessage, LLMResponse
from app.services.llm.token_counter import TokenCounter
from app.core.exceptions import ExternalServiceError


class OpenAIClient(LLMServiceBase):
    """
    OpenAI LLM service implementation.

    Supports GPT-3.5, GPT-4, and other OpenAI models.
    """

    # Available OpenAI models
    MODELS = [
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4-turbo-preview",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
    ]

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """
        Initialize OpenAI client.

        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-4-turbo-preview)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        super().__init__(api_key, model, max_tokens, temperature)

        # Initialize async OpenAI client
        self.client = AsyncOpenAI(api_key=api_key)

    async def generate(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a completion from OpenAI.

        Args:
            messages: List of conversation messages
            max_tokens: Override max tokens
            temperature: Override temperature
            **kwargs: Additional OpenAI parameters

        Returns:
            LLMResponse: Generated response

        Raises:
            ExternalServiceError: If API call fails
        """
        try:
            # Prepare parameters
            params = {
                "model": self.model,
                "messages": self._format_messages(messages),
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature,
            }

            # Add any additional parameters
            params.update(kwargs)

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            # Call OpenAI API
            response = await self.client.chat.completions.create(**params)

            # Extract response data
            choice = response.choices[0]
            content = choice.message.content
            finish_reason = choice.finish_reason
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

            return LLMResponse(
                content=content,
                model=response.model,
                finish_reason=finish_reason,
                usage=usage,
            )

        except Exception as e:
            raise ExternalServiceError(
                service="OpenAI",
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
        Generate a streaming completion from OpenAI.

        Args:
            messages: List of conversation messages
            max_tokens: Override max tokens
            temperature: Override temperature
            **kwargs: Additional OpenAI parameters

        Yields:
            str: Content chunks

        Raises:
            ExternalServiceError: If API call fails
        """
        try:
            # Prepare parameters
            params = {
                "model": self.model,
                "messages": self._format_messages(messages),
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature,
                "stream": True,
            }

            # Add any additional parameters
            params.update(kwargs)

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            # Call OpenAI API with streaming
            stream = await self.client.chat.completions.create(**params)

            # Yield chunks as they arrive
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            raise ExternalServiceError(
                service="OpenAI",
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
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            "gpt-4-turbo": 128000,
            "gpt-4-turbo-preview": 128000,
            "gpt-3.5-turbo": 4096,
            "gpt-3.5-turbo-16k": 16384,
        }

        return context_windows.get(self.model, 4096)
