"""
Unified graph retrieval service.

Combines Text-to-Cypher and graph embedding search via Reciprocal Rank Fusion,
mirroring the HybridSearchService pattern from the vector retrieval layer.
"""
import logging
from typing import Optional, List

from app.services.graph.base import GraphSearchResult
from app.services.graph.retrieval.text_to_cypher import TextToCypherService
from app.services.graph.retrieval.graph_embedding_search import GraphEmbeddingSearch

logger = logging.getLogger(__name__)


class GraphRetrievalService:
    """Unified graph retrieval: Text-to-Cypher + graph embedding via RRF."""

    def __init__(
        self,
        text_to_cypher: Optional[TextToCypherService] = None,
        graph_embedding_search: Optional[GraphEmbeddingSearch] = None,
        cypher_weight: float = 0.6,
        embedding_weight: float = 0.4,
    ) -> None:
        self._cypher = text_to_cypher
        self._embedding_search = graph_embedding_search
        self._cypher_weight = cypher_weight
        self._embedding_weight = embedding_weight

    async def search(
        self, query: str, top_k: int = 10, entity_hints: Optional[list] = None
    ) -> List[GraphSearchResult]:
        """Run both search modes and fuse via RRF."""
        cypher_results: List[GraphSearchResult] = []
        embedding_results: List[GraphSearchResult] = []

        if self._cypher:
            try:
                cypher_results = await self._cypher.query(query, max_results=top_k, entity_hints=entity_hints)
            except Exception as e:
                logger.warning("Text-to-Cypher search failed: %s", e)

        if self._embedding_search:
            try:
                embedding_results = await self._embedding_search.search(query, top_k=top_k)
            except Exception as e:
                logger.warning("Graph embedding search failed: %s", e)

        if not cypher_results and not embedding_results:
            return []

        if not cypher_results:
            return embedding_results[:top_k]

        if not embedding_results:
            return cypher_results[:top_k]

        return self._reciprocal_rank_fusion(
            cypher_results, embedding_results, top_k
        )

    def _reciprocal_rank_fusion(
        self,
        cypher_results: List[GraphSearchResult],
        embedding_results: List[GraphSearchResult],
        top_k: int,
        k: int = 60,
    ) -> List[GraphSearchResult]:
        """RRF fusion across two result sets, same algorithm as HybridSearchService."""
        scores: dict[str, float] = {}
        content_map: dict[str, GraphSearchResult] = {}

        for rank, result in enumerate(cypher_results):
            key = result.content[:200]
            scores[key] = scores.get(key, 0.0) + self._cypher_weight / (k + rank + 1)
            content_map[key] = result

        for rank, result in enumerate(embedding_results):
            key = result.content[:200]
            scores[key] = scores.get(key, 0.0) + self._embedding_weight / (k + rank + 1)
            if key not in content_map:
                content_map[key] = result

        sorted_keys = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]

        results: List[GraphSearchResult] = []
        for key in sorted_keys[:top_k]:
            result = content_map[key]
            results.append(
                GraphSearchResult(
                    content=result.content,
                    entities=result.entities,
                    relations=result.relations,
                    score=scores[key],
                    source_type=result.source_type,
                    metadata={
                        **(result.metadata or {}),
                        "rrf_score": scores[key],
                    },
                )
            )
        return results
