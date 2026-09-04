"""
Factory for creating LLM service instances.

Provides a simple interface for creating LLM clients based on configuration.
"""
from typing import Optional

from app.services.llm.base import LLMServiceBase
from app.services.llm.openai_client import OpenAIClient
from app.services.llm.anthropic_client import AnthropicClient
from app.services.llm.deepseek_client import DeepSeekClient
from app.config.settings import get_settings
from app.core.exceptions import ValidationError


class LLMFactory:
    """
    Factory for creating LLM service instances.

    Supports multiple LLM providers (DeepSeek, OpenAI, Anthropic) with unified interface.
    """

    SUPPORTED_PROVIDERS = ["deepseek", "openai", "anthropic"]

    @staticmethod
    def create(
        provider: str = "deepseek",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMServiceBase:
        """
        Create an LLM service instance for the specified provider.

        Args:
            provider: LLM provider name ("deepseek", "openai", "anthropic")
            api_key: API key (defaults to environment variable if not provided)
            model: Model name (defaults to provider default if not provided)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            LLMServiceBase: Configured LLM service instance

        Raises:
            ValidationError: If provider is not supported or configuration is invalid

        Examples:
            >>> # Create DeepSeek client with environment variables
            >>> llm = LLMFactory.create(provider="deepseek")
            >>>
            >>> # Create DeepSeek client with explicit API key
            >>> llm = LLMFactory.create(
            ...     provider="deepseek",
            ...     api_key="your-api-key",
            ...     model="deepseek-chat"
            ... )
            >>>
            >>> # Create OpenAI client
            >>> llm = LLMFactory.create(provider="openai")
        """
        settings = get_settings()

        # Validate provider
        provider = provider.lower()
        if provider not in LLMFactory.SUPPORTED_PROVIDERS:
            raise ValidationError(
                f"Unsupported LLM provider: {provider}. "
                f"Supported providers: {', '.join(LLMFactory.SUPPORTED_PROVIDERS)}"
            )

        # Create client based on provider
        if provider == "deepseek":
            api_key = api_key or settings.DEEPSEEK_API_KEY
            if not api_key:
                raise ValidationError("DeepSeek API key not configured")

            model = model or settings.DEEPSEEK_MODEL
            return DeepSeekClient(
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                base_url=settings.DEEPSEEK_BASE_URL,
            )

        elif provider == "openai":
            api_key = api_key or settings.OPENAI_API_KEY
            if not api_key:
                raise ValidationError("OpenAI API key not configured")

            model = model or settings.OPENAI_MODEL
            return OpenAIClient(
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        elif provider == "anthropic":
            api_key = api_key or settings.ANTHROPIC_API_KEY
            if not api_key:
                raise ValidationError("Anthropic API key not configured")

            model = model or settings.ANTHROPIC_MODEL
            return AnthropicClient(
                api_key=api_key,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        else:
            raise ValidationError(f"Provider {provider} not implemented")

    @staticmethod
    def create_from_settings(settings_override: Optional[dict] = None) -> LLMServiceBase:
        """
        Create LLM service from settings with automatic provider selection.

        Uses the first available API key in the order: DeepSeek, OpenAI, Anthropic.

        Args:
            settings_override: Optional settings dict to override defaults

        Returns:
            LLMServiceBase: Configured LLM service instance

        Raises:
            ValidationError: If no LLM provider is configured

        Examples:
            >>> # Auto-select provider (checks DeepSeek first, then OpenAI, then Anthropic)
            >>> llm = LLMFactory.create_from_settings()
        """
        settings = get_settings()

        # Check for DeepSeek first (highest priority for this user)
        if settings.DEEPSEEK_API_KEY:
            return LLMFactory.create(
                provider="deepseek",
                api_key=settings.DEEPSEEK_API_KEY,
                model=settings.DEEPSEEK_MODEL,
            )

        # Check for OpenAI
        if settings.OPENAI_API_KEY:
            return LLMFactory.create(
                provider="openai",
                api_key=settings.OPENAI_API_KEY,
                model=settings.OPENAI_MODEL,
            )

        # Check for Anthropic
        if settings.ANTHROPIC_API_KEY:
            return LLMFactory.create(
                provider="anthropic",
                api_key=settings.ANTHROPIC_API_KEY,
                model=settings.ANTHROPIC_MODEL,
            )

        raise ValidationError(
            "No LLM provider configured. Please set at least one of: "
            "DEEPSEEK_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY"
        )
