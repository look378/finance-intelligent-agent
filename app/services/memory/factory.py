"""
Factory for creating memory strategy instances.

Provides a simple interface for creating the appropriate memory strategy.
"""
from typing import Optional

from app.services.memory.base import MemoryStrategy
from app.services.memory.sliding_window import SlidingWindowMemory
from app.services.memory.summarization import SummarizationMemory
from app.services.memory.hybrid import HybridMemory
from app.services.memory.optimized_context import OptimizedContextBuilder
from app.repositories.message_repository import MessageRepository
from app.services.llm.base import LLMServiceBase
from app.services.embeddings import EmbeddingFactory
from app.core.exceptions import ValidationError


class MemoryFactory:
    """
    Factory for creating memory strategy instances.

    Provides a simple interface for creating memory strategies
    with proper configuration.
    """

    @staticmethod
    def create(
        memory_type: str,
        message_repo: MessageRepository,
        llm_service: Optional[LLMServiceBase] = None,
        **kwargs,
    ) -> MemoryStrategy:
        """
        Create a memory strategy instance.

        Args:
            memory_type: Type of memory strategy ("sliding_window", "summarization", "hybrid", "optimized")
            message_repo: Message repository instance
            llm_service: Optional LLM service (required for summarization/hybrid)
            **kwargs: Additional configuration for the strategy

        Returns:
            MemoryStrategy: Configured memory strategy instance

        Raises:
            ValidationError: If memory_type is invalid or required params missing
        """
        if memory_type == "sliding_window":
            # Extract sliding window parameters
            window_size = kwargs.get("window_size", 10)

            # Validate
            if not (1 <= window_size <= 100):
                raise ValidationError(
                    f"window_size must be between 1 and 100, got {window_size}"
                )

            return SlidingWindowMemory(
                message_repo=message_repo,
                window_size=window_size,
            )

        elif memory_type == "summarization":
            # Check required LLM service
            if llm_service is None:
                raise ValidationError(
                    "llm_service is required for summarization memory"
                )

            # Extract summarization parameters
            summary_threshold = kwargs.get("summary_threshold", 20)
            summary_interval = kwargs.get("summary_interval", 10)

            # Validate
            if summary_threshold < 5:
                raise ValidationError(
                    f"summary_threshold must be at least 5, got {summary_threshold}"
                )

            if summary_interval < 1:
                raise ValidationError(
                    f"summary_interval must be at least 1, got {summary_interval}"
                )

            return SummarizationMemory(
                message_repo=message_repo,
                llm_service=llm_service,
                summary_threshold=summary_threshold,
                summary_interval=summary_interval,
            )

        elif memory_type == "hybrid":
            # Check required LLM service
            if llm_service is None:
                raise ValidationError(
                    "llm_service is required for hybrid memory"
                )

            # Extract hybrid parameters
            window_size = kwargs.get("window_size", 10)
            summary_threshold = kwargs.get("summary_threshold", 20)
            summary_interval = kwargs.get("summary_interval", 10)
            hybrid_threshold = kwargs.get("hybrid_threshold", 30)

            # Validate
            if not (1 <= window_size <= 100):
                raise ValidationError(
                    f"window_size must be between 1 and 100, got {window_size}"
                )

            if summary_threshold < 5:
                raise ValidationError(
                    f"summary_threshold must be at least 5, got {summary_threshold}"
                )

            if hybrid_threshold < 5:
                raise ValidationError(
                    f"hybrid_threshold must be at least 5, got {hybrid_threshold}"
                )

            # Create sub-strategies
            sliding = SlidingWindowMemory(
                message_repo=message_repo,
                window_size=window_size,
            )

            summary = SummarizationMemory(
                message_repo=message_repo,
                llm_service=llm_service,
                summary_threshold=summary_threshold,
                summary_interval=summary_interval,
            )

            return HybridMemory(
                sliding_window=sliding,
                summarization=summary,
                hybrid_threshold=hybrid_threshold,
            )

        elif memory_type == "optimized":
            # Optimized context builder with relevance filtering
            embedding_service = kwargs.get("embedding_service")

            if embedding_service is None:
                # Create default embedding service
                embedding_service = EmbeddingFactory.create_from_settings()

            # Extract optimized parameters
            max_recent_messages = kwargs.get("max_recent_messages", 3)
            max_relevant_messages = kwargs.get("max_relevant_messages", 5)
            relevance_threshold = kwargs.get("relevance_threshold", 0.5)
            token_budget = kwargs.get("token_budget", 4096)

            # Validate
            if not (1 <= max_recent_messages <= 20):
                raise ValidationError(
                    f"max_recent_messages must be between 1 and 20, got {max_recent_messages}"
                )

            if not (1 <= max_relevant_messages <= 50):
                raise ValidationError(
                    f"max_relevant_messages must be between 1 and 50, got {max_relevant_messages}"
                )

            if not (0.0 <= relevance_threshold <= 1.0):
                raise ValidationError(
                    f"relevance_threshold must be between 0.0 and 1.0, got {relevance_threshold}"
                )

            if token_budget < 512:
                raise ValidationError(
                    f"token_budget must be at least 512, got {token_budget}"
                )

            return OptimizedContextBuilder(
                message_repo=message_repo,
                embedding_service=embedding_service,
                max_recent_messages=max_recent_messages,
                max_relevant_messages=max_relevant_messages,
                relevance_threshold=relevance_threshold,
                token_budget=token_budget,
            )

        else:
            # Invalid memory type
            valid_types = ["sliding_window", "summarization", "hybrid", "optimized"]
            raise ValidationError(
                f"Invalid memory_type: {memory_type}. Must be one of {valid_types}"
            )
