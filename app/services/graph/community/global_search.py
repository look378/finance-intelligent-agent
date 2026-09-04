"""
Global search service.

Answers broad/global queries using community summaries rather than
individual entity lookups. Suitable for questions like "总结整个固收+理财产品的市场格局".
"""
import logging
from typing import List

from app.services.graph.base import GraphClient, GraphSearchResult
from app.services.embeddings.base import EmbeddingServiceBase

logger = logging.getLogger(__name__)


class GlobalSearchService:
    """Answer global queries using community summaries."""

    def __init__(
        self,
        graph_client: GraphClient,
        embedding_service: EmbeddingServiceBase,
    ) -> None:
        self._graph = graph_client
        self._embeddings = embedding_service

    async def search(
        self, query: str, top_k: int = 5
    ) -> List[GraphSearchResult]:
        """Search community summaries by embedding similarity."""
        query_embedding = await self._embeddings.embed_single(query)

        # Try vector index on CommunitySummary nodes
        try:
            results = await self._graph.execute_cypher(
                "CALL db.index.vector.queryNodes('community_summary_embedding', $k, $embedding) "
                "YIELD node, score "
                "RETURN node.community_id AS cid, node.title AS title, "
                "node.summary AS summary, node.entity_count AS entity_count, "
                "node.entities AS entities, score "
                "ORDER BY score DESC",
                {"k": top_k, "embedding": query_embedding},
            )
        except Exception:
            # Fallback: scan all CommunitySummary nodes
            results = await self._fallback_search(query_embedding, top_k)

        search_results: List[GraphSearchResult] = []
        for r in results:
            content = r.get("summary", "")
            if not content:
                continue

            title = r.get("title", "")
            entities_list = r.get("entities", [])

            search_results.append(
                GraphSearchResult(
                    content=f"{title}\n\n{content}",
                    entities=[],
                    relations=[],
                    score=r.get("score", 0.0),
                    source_type="community_summary",
                    metadata={
                        "community_id": r.get("cid"),
                        "entity_count": r.get("entity_count", 0),
                        "entities": entities_list if isinstance(entities_list, list) else [],
                    },
                )
            )

        return search_results[:top_k]

    async def _fallback_search(
        self, query_embedding: List[float], top_k: int
    ) -> List[dict]:
        """Fallback when vector index is not available."""
        try:
            return await self._graph.execute_cypher(
                "MATCH (cs:CommunitySummary) "
                "WHERE cs.embedding IS NOT NULL "
                "RETURN cs.community_id AS cid, cs.title AS title, "
                "cs.summary AS summary, cs.entity_count AS entity_count, "
                "cs.entities AS entities, 0.5 AS score "
                "LIMIT $k",
                {"k": top_k},
            )
        except Exception:
            return []
