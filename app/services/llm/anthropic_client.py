"""
Anthropic LLM client implementation.

Provides integration with Anthropic's Claude models.
"""
from typing import AsyncGenerator, Optional, List

from anthropic import AsyncAnthropic

from app.services.llm.base import LLMServiceBase, LLMMessage, LLMResponse
from app.services.llm.token_counter import TokenCounter
from app.core.exceptions import ExternalServiceError


class AnthropicClient(LLMServiceBase):
    """
    Anthropic LLM service implementation.

    Supports Claude 3 models (Opus, Sonnet, Haiku).
    """

    # Available Anthropic models
    MODELS = [
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "claude-3-opus-20240229@KLIENT_ALTERNAT_LOCATION",  # For AWS Bedrock
    ]

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-opus-20240229",
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> None:
        """
        Initialize Anthropic client.

        Args:
            api_key: Anthropic API key
            model: Model name (default: claude-3-opus-20240229)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        """
        super().__init__(api_key, model, max_tokens, temperature)

        # Initialize async Anthropic client
        self.client = AsyncAnthropic(api_key=api_key)

    async def generate(
        self,
        messages: List[LLMMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate a completion from Anthropic.

        Args:
            messages: List of conversation messages
            max_tokens: Override max tokens
            temperature: Override temperature
            **kwargs: Additional Anthropic parameters

        Returns:
            LLMResponse: Generated response

        Raises:
            ExternalServiceError: If API call fails
        """
        try:
            # Extract system message if present
            system_message = None
            conversation_messages = []

            for msg in messages:
                if msg.role == "system":
                    system_message = msg.content
                else:
                    # Convert role format if needed
                    # Anthropic uses "user" and "assistant" (no "system" role)
                    conversation_messages.append(msg.to_dict())

            # Prepare parameters
            params = {
                "model": self.model,
                "messages": conversation_messages,
                "max_tokens": max_tokens or self.max_tokens or 4096,
                "temperature": temperature or self.temperature,
            }

            # Add system message if present
            if system_message:
                params["system"] = system_message

            # Add any additional parameters
            params.update(kwargs)

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            # Call Anthropic API
            response = await self.client.messages.create(**params)

            # Extract response data
            # Anthropic returns content blocks
            content_blocks = response.content
            content = "".join(
                block.text for block in content_blocks if block.type == "text"
            )

            finish_reason = response.stop_reason
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

            return LLMResponse(
                content=content,
                model=response.model,
                finish_reason=finish_reason,
                usage=usage,
            )

        except Exception as e:
            raise ExternalServiceError(
                service="Anthropic",
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
        Generate a streaming completion from Anthropic.

        Args:
            messages: List of conversation messages
            max_tokens: Override max tokens
            temperature: Override temperature
            **kwargs: Additional Anthropic parameters

        Yields:
            str: Content chunks

        Raises:
            ExternalServiceError: If API call fails
        """
        try:
            # Extract system message if present
            system_message = None
            conversation_messages = []

            for msg in messages:
                if msg.role == "system":
                    system_message = msg.content
                else:
                    conversation_messages.append(msg.to_dict())

            # Prepare parameters
            params = {
                "model": self.model,
                "messages": conversation_messages,
                "max_tokens": max_tokens or self.max_tokens or 4096,
                "temperature": temperature or self.temperature,
            }

            # Add system message if present
            if system_message:
                params["system"] = system_message

            # Add any additional parameters
            params.update(kwargs)

            # Remove None values
            params = {k: v for k, v in params.items() if v is not None}

            # Call Anthropic API with streaming
            async with self.client.messages.stream(**params) as stream:
                # Yield text chunks as they arrive
                async for event in stream:
                    if event.type == "content_block_delta":
                        if hasattr(event.delta, "text"):
                            yield event.delta.text

        except Exception as e:
            raise ExternalServiceError(
                service="Anthropic",
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

        Note: Anthropic doesn't provide a public tokenizer,
        so we use tiktoken as an approximation.

        Args:
            messages: List of messages

        Returns:
            int: Total token count
        """
        return TokenCounter.count_messages(messages, "claude-3")

    def get_max_context_tokens(self) -> int:
        """
        Get the maximum context window for the current model.

        Returns:
            int: Maximum tokens in context window
        """
        context_windows = {
            "claude-3-opus-20240229": 200000,
            "claude-3-sonnet-20240229": 200000,
            "claude-3-haiku-20240307": 200000,
        }

        return context_windows.get(self.model, 200000)
