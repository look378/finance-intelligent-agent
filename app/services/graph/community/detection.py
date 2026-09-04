"""
Community detection service.

Detects communities (clusters) in the knowledge graph using either Neo4j GDS
Leiden algorithm or a pure Python fallback.
"""
import logging
from typing import Dict, List, Any

from app.services.graph.base import (
    GraphClient,
    CommunitySummary,
    GraphClientError,
)

logger = logging.getLogger(__name__)


class CommunityDetectionService:
    """Run community detection on the knowledge graph."""

    def __init__(self, graph_client: GraphClient) -> None:
        self._graph = graph_client

    async def detect_communities(
        self,
        min_community_size: int = 3,
        max_levels: int = 5,
    ) -> Dict[int, List[CommunitySummary]]:
        """Detect communities and return hierarchical summaries.

        Strategy:
        1. Try Neo4j GDS Leiden algorithm
        2. Fallback to pure Python using networkx
        """
        try:
            return await self._detect_with_gds(min_community_size)
        except Exception as e:
            logger.info("GDS detection failed (%s), trying Python fallback", e)
            return await self._detect_with_python(min_community_size)

    async def _detect_with_gds(
        self, min_community_size: int
    ) -> Dict[int, List[CommunitySummary]]:
        """Use Neo4j GDS Leiden algorithm."""
        # Project graph
        await self._graph.execute_cypher(
            "CALL gds.graph.project('entity_graph', 'Entity', {"
            "  ALL: {orientation: 'UNDIRECTED'}"
            "})",
        )

        try:
            # Run Leiden
            await self._graph.execute_cypher(
                "CALL gds.leiden.write('entity_graph', {"
                f"  writeProperty: 'communityId',"
                f"  minCommunitySize: {min_community_size}"
                "})",
            )

            # Read communities
            results = await self._graph.execute_cypher(
                "MATCH (e:Entity) "
                "WHERE e.communityId IS NOT NULL "
                "RETURN e.communityId AS cid, collect(e.name) AS members, "
                "count(e) AS entity_count "
                "ORDER BY entity_count DESC"
            )
        finally:
            # Clean up projection
            try:
                await self._graph.execute_cypher(
                    "CALL gds.graph.drop('entity_graph')"
                )
            except Exception:
                pass

        communities: Dict[int, List[CommunitySummary]] = {0: []}
        for r in results:
            communities[0].append(
                CommunitySummary(
                    community_id=str(r["cid"]),
                    level=0,
                    title=f"Community {r['cid']}",
                    summary="",
                    entity_count=r["entity_count"],
                    entities=r["members"],
                )
            )
        return communities

    async def _detect_with_python(
        self, min_community_size: int
    ) -> Dict[int, List[CommunitySummary]]:
        """Pure Python fallback using networkx + python-louvain."""
        try:
            import networkx as nx
        except ImportError:
            logger.warning(
                "networkx not installed. Install with: pip install networkx python-louvain"
            )
            return {}

        # Export graph from Neo4j
        edges = await self._graph.execute_cypher(
            "MATCH (a:Entity)-[r]->(b:Entity) "
            "RETURN a.name AS source, b.name AS target"
        )

        nodes = await self._graph.execute_cypher(
            "MATCH (e:Entity) RETURN e.name AS name, e.type AS type"
        )

        if not nodes:
            return {}

        # Build networkx graph
        G = nx.Graph()
        for node in nodes:
            G.add_node(node["name"], type=node["type"])
        for edge in edges:
            G.add_edge(edge["source"], edge["target"])

        if G.number_of_edges() == 0:
            return {}

        # Run community detection
        try:
            from community import best_partition

            partition = best_partition(G)
        except ImportError:
            logger.warning("python-louvain not installed. Using connected components.")
            partition = {}
            for i, component in enumerate(nx.connected_components(G)):
                for node in component:
                    partition[node] = i

        # Group by community
        community_groups: Dict[int, List[str]] = {}
        for node, comm_id in partition.items():
            community_groups.setdefault(comm_id, []).append(node)

        communities: Dict[int, List[CommunitySummary]] = {0: []}
        for comm_id, members in community_groups.items():
            if len(members) < min_community_size:
                continue
            communities[0].append(
                CommunitySummary(
                    community_id=str(comm_id),
                    level=0,
                    title=f"Community {comm_id}",
                    summary="",
                    entity_count=len(members),
                    entities=members,
                )
            )

        # Write community IDs back to Neo4j
        for comm_id, members in community_groups.items():
            node_data = [{"name": m, "cid": str(comm_id)} for m in members]
            for nd in node_data:
                try:
                    await self._graph.execute_cypher(
                        "MATCH (e:Entity {name: $name}) SET e.communityId = $cid",
                        nd,
                    )
                except GraphClientError:
                    pass  # Read-only validation in execute_cypher; use direct driver

        return communities
