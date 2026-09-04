"""
Base embedding service interface and data structures.

Provides abstract interfaces for embedding providers to implement.
"""
from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass


@dataclass
class EmbeddingResult:
    """
    Result of an embedding generation operation.

    Attributes:
        embeddings: List of embedding vectors (list of floats)
        model: Model name/identifier used
        dimensions: Number of dimensions per vector
        tokens_used: Total tokens used for embedding generation
    """
    embeddings: List[List[float]]
    model: str
    dimensions: int
    tokens_used: int


class EmbeddingServiceBase(ABC):
    """
    Abstract base class for embedding service implementations.

    All embedding provider implementations (Local, OpenAI, etc.) must
    inherit from this class and implement the required methods.
    """

    def __init__(
        self,
        model: str,
        dimensions: int,
    ) -> None:
        """
        Initialize the embedding service.

        Args:
            model: Model name/identifier
            dimensions: Embedding vector dimensions
        """
        self.model = model
        self.dimensions = dimensions

    @abstractmethod
    async def embed(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of text strings to embed

        Returns:
            EmbeddingResult: Generated embeddings with metadata

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("embed() must be implemented by subclass")

    @abstractmethod
    async def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text string to embed

        Returns:
            List[float]: Embedding vector

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("embed_single() must be implemented by subclass")

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in text.

        Args:
            text: Text to estimate tokens for

        Returns:
            int: Estimated token count (rough estimate: chars / 4)
        """
        return len(text) // 4

    def get_batch_size(self) -> int:
        """
        Get the recommended batch size for embedding generation.

        Returns:
            int: Recommended batch size
        """
        return 32  # Default batch size
