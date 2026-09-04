"""
Graph embedding search.

Searches graph entities by embedding similarity and expands to
subgraph context for richer results.
"""
import logging
from typing import Optional, List

from app.services.graph.base import (
    GraphClient,
    GraphEntity,
    GraphSearchResult,
)
from app.services.embeddings.base import EmbeddingServiceBase

logger = logging.getLogger(__name__)


class GraphEmbeddingSearch:
    """Search graph entities by embedding, expand to neighborhood context."""

    def __init__(
        self,
        graph_client: GraphClient,
        embedding_service: EmbeddingServiceBase,
        max_hops: int = 1,
    ) -> None:
        self._graph = graph_client
        self._embeddings = embedding_service
        self._max_hops = max_hops

    async def search(
        self,
        query: str,
        top_k: int = 10,
        entity_types: Optional[List[str]] = None,
    ) -> List[GraphSearchResult]:
        """Embed query, find similar entities, expand to neighborhoods."""
        query_embedding = await self._embeddings.embed_single(query)

        entity_results = await self._graph.search_entities_by_embedding(
            query_embedding, top_k=top_k, entity_types=entity_types
        )

        if not entity_results:
            return []

        expanded: List[GraphSearchResult] = []
        seen_names: set[str] = set()

        for result in entity_results:
            if not result.entities:
                continue

            entity = result.entities[0]
            if entity.name in seen_names:
                continue
            seen_names.add(entity.name)

            # Get 1-hop neighborhood for context
            try:
                neighborhood = await self._graph.get_entity_neighborhood(
                    entity_name=entity.name,
                    max_hops=self._max_hops,
                    limit=10,
                )
            except Exception as e:
                logger.debug("Neighborhood expansion failed for %s: %s", entity.name, e)
                neighborhood = []

            context_parts: List[str] = []
            if entity.description:
                context_parts.append(entity.description)

            all_entities = [entity]
            all_relations = []

            for nb in neighborhood:
                context_parts.append(nb.content)
                all_entities.extend(nb.entities)
                all_relations.extend(nb.relations)

            expanded.append(
                GraphSearchResult(
                    content=" | ".join(context_parts),
                    entities=list({e.name: e for e in all_entities}.values()),
                    relations=all_relations,
                    score=result.score,
                    source_type="graph_embedding",
                    metadata={"entity_type": entity.type},
                )
            )

        return expanded[:top_k]
