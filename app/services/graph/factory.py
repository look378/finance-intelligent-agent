"""
Factory for creating graph database clients.

Follows the same pattern as RetrievalFactory.
"""
from typing import Optional

from app.services.graph.base import GraphClient
from app.core.exceptions import ValidationError


class GraphFactory:
    """Factory for creating graph database client instances."""

    @staticmethod
    def create_client(
        client_type: str = "neo4j",
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        max_connection_pool_size: int = 50,
        connection_timeout: float = 30.0,
    ) -> GraphClient:
        if client_type == "neo4j":
            from app.services.graph.neo4j_client import Neo4jClient

            return Neo4jClient(
                uri=uri or "bolt://localhost:7687",
                user=user or "neo4j",
                password=password or "password",
                database=database or "neo4j",
                max_connection_pool_size=max_connection_pool_size,
                connection_timeout=connection_timeout,
            )
        else:
            raise ValidationError(
                f"Invalid client_type: {client_type}. Must be one of ['neo4j']"
            )

    @staticmethod
    def create_from_settings() -> Optional[GraphClient]:
        """Create from app settings. Returns None if GRAPH_RAG_ENABLED=False."""
        from app.config.settings import get_settings

        settings = get_settings()

        if not settings.GRAPH_RAG_ENABLED:
            return None

        return GraphFactory.create_client(
            client_type="neo4j",
            uri=settings.NEO4J_URI,
            user=settings.NEO4J_USER,
            password=settings.NEO4J_PASSWORD,
            database=settings.NEO4J_DATABASE,
            max_connection_pool_size=settings.NEO4J_MAX_CONNECTION_POOL_SIZE,
            connection_timeout=settings.NEO4J_CONNECTION_TIMEOUT,
        )
