"""
Embedding services package.

Exports all embedding-related components including providers and factory.
"""
from app.services.embeddings.base import (
    EmbeddingServiceBase,
    EmbeddingResult
)
from app.services.embeddings.local_embeddings import LocalEmbeddingService
from app.services.embeddings.cached_embeddings import CachedEmbeddingService
from app.services.embeddings.factory import EmbeddingFactory

__all__ = [
    # Base classes
    "EmbeddingServiceBase",
    "EmbeddingResult",
    # Providers
    "LocalEmbeddingService",
    # Caching
    "CachedEmbeddingService",
    # Factory
    "EmbeddingFactory",
]
