"""
Graph service base interface and data models.

Provides abstract interface for graph database operations, mirroring the
VectorClient ABC pattern from app/services/retrieval/vector_base.py.
"""
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


@dataclass
class GraphEntity:
    """An entity node in the knowledge graph."""

    id: str
    name: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    embedding: Optional[List[float]] = None


@dataclass
class GraphRelation:
    """A relationship (edge) in the knowledge graph."""

    id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None


@dataclass
class GraphSearchResult:
    """Result from a graph query, analogous to SearchResult."""

    content: str
    entities: List[GraphEntity]
    relations: List[GraphRelation]
    score: float
    source_type: str  # "text_to_cypher" | "graph_embedding" | "community_summary"
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CommunitySummary:
    """A community (cluster) summary for global context."""

    community_id: str
    level: int
    title: str
    summary: str
    entity_count: int
    entities: List[str]
    embedding: Optional[List[float]] = None


class GraphClient(ABC):
    """Abstract base class for graph database clients."""

    @abstractmethod
    async def connect(self) -> None:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        pass

    @abstractmethod
    async def add_entities(self, entities: List[GraphEntity]) -> List[str]:
        pass

    @abstractmethod
    async def add_relations(self, relations: List[GraphRelation]) -> List[str]:
        pass

    @abstractmethod
    async def execute_cypher(
        self, query: str, params: Optional[dict] = None
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def search_entities_by_embedding(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        entity_types: Optional[List[str]] = None,
    ) -> List[GraphSearchResult]:
        pass

    @abstractmethod
    async def get_entity_neighborhood(
        self, entity_name: str, max_hops: int = 2, limit: int = 50
    ) -> List[GraphSearchResult]:
        pass

    @abstractmethod
    async def get_schema(self) -> str:
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, int]:
        pass


class GraphClientError(Exception):
    """Base exception for graph client errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
