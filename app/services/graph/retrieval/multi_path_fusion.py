"""
Multi-path retrieval fusion.

Fuses results from the vector retrieval path (Qdrant) and the graph
retrieval path (Neo4j) into a unified result list via RRF.
"""
import logging
from typing import List

from app.services.retrieval.vector_base import SearchResult
from app.services.graph.base import GraphSearchResult

logger = logging.getLogger(__name__)


class MultiPathRetrievalFusion:
    """Fuse vector and graph retrieval results via RRF."""

    def __init__(
        self,
        vector_weight: float = 0.5,
        graph_weight: float = 0.5,
    ) -> None:
        self._vector_weight = vector_weight
        self._graph_weight = graph_weight

    def fuse(
        self,
        vector_results: List[SearchResult],
        graph_results: List[GraphSearchResult],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Fuse vector + graph results into unified SearchResult list."""
        scores: dict[str, float] = {}
        result_map: dict[str, SearchResult] = {}

        for rank, result in enumerate(vector_results):
            key = result.document_id
            scores[key] = scores.get(key, 0.0) + self._vector_weight / (60 + rank + 1)
            result_map[key] = result

        for rank, graph_result in enumerate(graph_results):
            key = f"graph_{hash(graph_result.content[:200])}"
            score = self._graph_weight / (60 + rank + 1)
            scores[key] = scores.get(key, 0.0) + score

            if key not in result_map:
                result_map[key] = SearchResult(
                    document_id=key,
                    content=graph_result.content,
                    score=0.0,
                    metadata={
                        "source_type": graph_result.source_type,
                        "entities": [
                            {"name": e.name, "type": e.type}
                            for e in graph_result.entities
                        ],
                        "relations": [
                            {"type": r.relation_type}
                            for r in graph_result.relations
                        ],
                    },
                )

        sorted_keys = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]

        fused: List[SearchResult] = []
        for key in sorted_keys[:top_k]:
            result = result_map[key]
            fused.append(
                SearchResult(
                    document_id=result.document_id,
                    content=result.content,
                    score=scores[key],
                    metadata={
                        **(result.metadata or {}),
                        "rrf_score": scores[key],
                    },
                )
            )
        return fused
