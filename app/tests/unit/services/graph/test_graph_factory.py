"""Unit tests for graph factory."""
import pytest
from unittest.mock import patch, MagicMock

from app.services.graph.factory import GraphFactory
from app.services.graph.neo4j_client import Neo4jClient
from app.core.exceptions import ValidationError


class TestGraphFactory:
    def test_create_neo4j_client(self):
        client = GraphFactory.create_client(
            client_type="neo4j",
            uri="bolt://test:7687",
            user="neo4j",
            password="pass",
        )
        assert isinstance(client, Neo4jClient)

    def test_create_invalid_type_raises(self):
        with pytest.raises(ValidationError, match="Invalid client_type"):
            GraphFactory.create_client(client_type="redis")

    @patch("app.config.settings.get_settings")
    def test_create_from_settings_disabled(self, mock_settings):
        mock = MagicMock()
        mock.GRAPH_RAG_ENABLED = False
        mock_settings.return_value = mock
        assert GraphFactory.create_from_settings() is None

    @patch("app.config.settings.get_settings")
    def test_create_from_settings_enabled(self, mock_settings):
        mock = MagicMock()
        mock.GRAPH_RAG_ENABLED = True
        mock.NEO4J_URI = "bolt://localhost:7687"
        mock.NEO4J_USER = "neo4j"
        mock.NEO4J_PASSWORD = "pass"
        mock.NEO4J_DATABASE = "neo4j"
        mock.NEO4J_MAX_CONNECTION_POOL_SIZE = 50
        mock.NEO4J_CONNECTION_TIMEOUT = 30.0
        mock_settings.return_value = mock

        client = GraphFactory.create_from_settings()
        assert isinstance(client, Neo4jClient)
