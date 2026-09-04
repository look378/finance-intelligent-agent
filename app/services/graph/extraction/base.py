"""
Entity extraction base interface and data models.
"""
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


@dataclass
class ExtractionResult:
    """Result of entity/relationship extraction from text."""

    entities: List[Dict[str, Any]] = field(default_factory=list)
    relations: List[Dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""


class EntityExtractor(ABC):
    """Abstract base class for entity extractors."""

    @abstractmethod
    async def extract(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> ExtractionResult:
        """Extract entities and relations from text."""
        pass

    @abstractmethod
    async def extract_batch(
        self, texts: List[str], context: Optional[Dict[str, Any]] = None
    ) -> List[ExtractionResult]:
        """Extract entities and relations from multiple texts."""
        pass
