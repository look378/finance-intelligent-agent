"""
Base interface for intent detection.

Provides abstract classes and data structures for intent classifiers.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any

from app.models.enums.intent import Intent


@dataclass
class IntentResult:
    """
    Result of intent detection.

    Attributes:
        intent: Detected intent category
        confidence: Confidence score (0.0 to 1.0)
        metadata: Additional metadata about detection
    """
    intent: Intent
    confidence: float = 0.0
    metadata: Optional[dict] = None


class IntentDetector(ABC):
    """
    Abstract base class for intent detectors.

    All intent detection implementations must inherit from this class
    and implement the required methods.
    """

    @abstractmethod
    def detect(self, query: str, context: Optional[dict] = None) -> Intent:
        """
        Detect the intent of a user query.

        Args:
            query: User's text query
            context: Optional context information (conversation history, etc.)

        Returns:
            Intent: Detected intent category

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("detect() must be implemented by subclass")

    @abstractmethod
    def detect_with_confidence(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> IntentResult:
        """
        Detect intent and return confidence score.

        Args:
            query: User's text query
            context: Optional context information

        Returns:
            IntentResult: Detected intent with confidence score

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("detect_with_confidence() must be implemented by subclass")

    def _normalize_query(self, query: str) -> str:
        """
        Normalize query text for processing.

        Args:
            query: Query text

        Returns:
            str: Normalized query
        """
        if not query:
            return ""

        # Convert to lowercase
        normalized = query.lower().strip()

        # Remove extra whitespace
        import re
        normalized = re.sub(r"\s+", " ", normalized)

        return normalized

    def _contains_any(self, text: str, keywords: List[str]) -> bool:
        """
        Check if text contains any of the keywords.

        Args:
            text: Text to search
            keywords: List of keywords to search for

        Returns:
            bool: True if any keyword is found
        """
        return any(keyword in text for keyword in keywords)

    def _count_matches(self, text: str, keywords: List[str]) -> int:
        """
        Count how many keywords appear in text.

        Args:
            text: Text to search
            keywords: List of keywords to search for

        Returns:
            int: Number of keyword matches
        """
        return sum(1 for keyword in keywords if keyword in text)
