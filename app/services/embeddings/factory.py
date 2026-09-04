"""
Factory for creating embedding service instances.

Provides a simple interface for creating embedding clients based on configuration.
"""
from typing import Optional

from app.services.embeddings.base import EmbeddingServiceBase
from app.services.embeddings.local_embeddings import LocalEmbeddingService
from app.services.embeddings.cached_embeddings import CachedEmbeddingService
from app.config.settings import get_settings
from app.core.exceptions import ValidationError


class EmbeddingFactory:
    """
    Factory for creating embedding service instances.

    Supports multiple embedding providers (Local, OpenAI) with unified interface.
    """

    SUPPORTED_PROVIDERS = ["local", "openai"]

    @staticmethod
    def create(
        provider: str = "local",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        use_cache: bool = True,
        cache_ttl: int = 604800,  # 7 days
        device: str = "cpu",
    ) -> EmbeddingServiceBase:
        """
        Create an embedding service instance for the specified provider.

        Args:
            provider: Embedding provider name ("local", "openai")
            model: Model name (defaults to provider default if not provided)
            api_key: API key (defaults to environment variable if not provided)
            use_cache: Whether to wrap service with Redis caching
            cache_ttl: Cache TTL in seconds (if use_cache=True)
            device: Device for local models ("cpu" or "cuda")

        Returns:
            EmbeddingServiceBase: Configured embedding service instance

        Raises:
            ValidationError: If provider is not supported or configuration is invalid

        Examples:
            >>> # Create local BGE-M3 embeddings (default)
            >>> embeddings = EmbeddingFactory.create(provider="local")
            >>>
            >>> # Create local embeddings without cache
            >>> embeddings = EmbeddingFactory.create(
            ...     provider="local",
            ...     use_cache=False
            ... )
        """
        settings = get_settings()

        # Validate provider
        provider = provider.lower()
        if provider not in EmbeddingFactory.SUPPORTED_PROVIDERS:
            raise ValidationError(
                f"Unsupported embedding provider: {provider}. "
                f"Supported providers: {', '.join(EmbeddingFactory.SUPPORTED_PROVIDERS)}"
            )

        # Create service based on provider
        if provider == "local":
            model = model or getattr(settings, 'EMBEDDING_MODEL', 'bge-m3-v2-zh')
            service = LocalEmbeddingService(
                model=model,
                device=device
            )

        elif provider == "openai":
            api_key = api_key or settings.OPENAI_API_KEY
            if not api_key:
                raise ValidationError("OpenAI API key not configured")

            # OpenAI embeddings (if needed later)
            from app.services.embeddings.openai_embeddings import OpenAIEmbeddingService
            model = model or settings.OPENAI_EMBEDDING_MODEL
            service = OpenAIEmbeddingService(
                api_key=api_key,
                model=model
            )

        else:
            raise ValidationError(f"Provider {provider} not implemented")

        # Wrap with cache if requested
        if use_cache:
            redis_url = settings.REDIS_URL
            service = CachedEmbeddingService(
                embedding_service=service,
                redis_url=redis_url,
                cache_ttl=cache_ttl
            )

        return service

    @staticmethod
    def create_from_settings(
        use_cache: bool = True,
        cache_ttl: int = 604800
    ) -> EmbeddingServiceBase:
        """
        Create embedding service from settings with automatic provider selection.

        Uses the provider specified in settings, or defaults to local embeddings.

        Args:
            use_cache: Whether to wrap service with Redis caching
            cache_ttl: Cache TTL in seconds

        Returns:
            EmbeddingServiceBase: Configured embedding service instance

        Raises:
            ValidationError: If no embedding provider is configured

        Examples:
            >>> # Auto-select from settings
            >>> embeddings = EmbeddingFactory.create_from_settings()
        """
        settings = get_settings()

        # Get provider from settings (default to local)
        provider = getattr(settings, 'EMBEDDING_PROVIDER', 'local')
        model = getattr(settings, 'EMBEDDING_MODEL', None)

        return EmbeddingFactory.create(
            provider=provider,
            model=model,
            use_cache=use_cache,
            cache_ttl=cache_ttl
        )
