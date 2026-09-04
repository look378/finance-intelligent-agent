"""Unit tests for Neo4j client with mocked driver."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.graph.neo4j_client import Neo4jClient, _validate_read_only, _sanitize_label
from app.services.graph.base import GraphEntity, GraphClientError


class TestNeo4jClient:
    def setup_method(self):
        self.client = Neo4jClient(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="test",
            database="neo4j",
        )
        self.client._driver = MagicMock()

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        self.client._driver.verify_connectivity = AsyncMock(return_value=None)
        assert await self.client.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_no_driver(self):
        self.client._driver = None
        assert await self.client.health_check() is False

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        self.client._driver.verify_connectivity = AsyncMock(side_effect=Exception("no connection"))
        assert await self.client.health_check() is False

    @pytest.mark.asyncio
    async def test_add_entities(self):
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=[{"id": "e1"}])
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        self.client._driver.session = MagicMock(return_value=mock_session)

        entities = [
            GraphEntity(id="e1", name="安心货币A", type="FinancialProduct", description="货币基金，风险等级R1"),
        ]
        ids = await self.client.add_entities(entities)
        assert "e1" in ids

    @pytest.mark.asyncio
    async def test_add_entities_empty(self):
        ids = await self.client.add_entities([])
        assert ids == []

    @pytest.mark.asyncio
    async def test_add_relations(self):
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=[{"rel_type": "RESOLVES"}])
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        self.client._driver.session = MagicMock(return_value=mock_session)

        from app.services.graph.base import GraphRelation
        relations = [
            GraphRelation(id="r1", source_entity_id="e1", target_entity_id="e2", relation_type="RESOLVES"),
        ]
        ids = await self.client.add_relations(relations)
        assert "r1" in ids

    @pytest.mark.asyncio
    async def test_execute_cypher_validates_read_only(self):
        with pytest.raises(GraphClientError, match="Only read-only"):
            await self.client.execute_cypher("CREATE (n:Test)")

    @pytest.mark.asyncio
    async def test_get_stats(self):
        mock_session = AsyncMock()
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=[{"count": 42}])
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        self.client._driver.session = MagicMock(return_value=mock_session)

        stats = await self.client.get_stats()
        assert stats["entities"] == 42
        assert stats["relations"] == 42


class TestValidateReadOnly:
    def test_match_query_passes(self):
        _validate_read_only("MATCH (n) RETURN n")

    def test_create_query_fails(self):
        with pytest.raises(GraphClientError, match="read-only"):
            _validate_read_only("CREATE (n:Test)")

    def test_delete_query_fails(self):
        with pytest.raises(GraphClientError, match="read-only"):
            _validate_read_only("MATCH (n) DELETE n")

    def test_merge_query_fails(self):
        with pytest.raises(GraphClientError, match="read-only"):
            _validate_read_only("MERGE (n:Test {name: 'x'})")

    def test_set_newline_bypass_blocked(self):
        """换行绕过（SET\\n）必须被拦截（旧黑名单可绕过）。"""
        with pytest.raises(GraphClientError, match="read-only"):
            _validate_read_only("MATCH (n) SET\n n.x = 1 RETURN n")

    def test_write_procedure_blocked(self):
        """写过程调用必须被拦截。"""
        with pytest.raises(GraphClientError, match="read-only"):
            _validate_read_only("CALL gds.leiden.write('g', {}) YIELD communityId")

    def test_readonly_relation_pattern_passes(self):
        """关系模式 -[:BELONGS_TO]-> 不应被误判为写操作。"""
        _validate_read_only(
            "MATCH (f:Entity:Fund)-[:BELONGS_TO]->(r) RETURN f.fund_name, r.level"
        )

    def test_as_alias_passes(self):
        """AS 别名（RETURN x AS create）不应被误判为写操作。"""
        _validate_read_only("MATCH (n) RETURN n.name AS create")


class TestSanitizeLabel:
    def test_valid_label(self):
        assert _sanitize_label("RESOLVES") == "RESOLVES"

    def test_strips_special_chars(self):
        assert _sanitize_label("HAS-FEATURE") == "HASFEATURE"

    def test_empty_label_raises(self):
        with pytest.raises(GraphClientError, match="Invalid"):
            _sanitize_label("---")
