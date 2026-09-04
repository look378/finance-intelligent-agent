"""Unit tests for graph retrieval services."""
import pytest
from unittest.mock import AsyncMock

from app.services.graph.base import GraphEntity, GraphRelation, GraphSearchResult
from app.services.graph.retrieval.graph_retrieval_service import GraphRetrievalService
from app.services.graph.retrieval.multi_path_fusion import MultiPathRetrievalFusion
from app.services.retrieval.vector_base import SearchResult


class TestGraphRetrievalServiceRRF:
    @pytest.mark.asyncio
    async def test_fuses_cypher_and_embedding_results(self):
        cypher_result = GraphSearchResult(
            content="Cypher: 安心货币A属于R1风险等级",
            entities=[GraphEntity(id="1", name="安心货币A", type="FinancialProduct")],
            relations=[],
            score=1.0,
            source_type="text_to_cypher",
        )
        embedding_result = GraphSearchResult(
            content="Embedding: 安心货币A的产品说明",
            entities=[GraphEntity(id="2", name="安心货币A", type="FinancialProduct")],
            relations=[],
            score=0.9,
            source_type="graph_embedding",
        )

        mock_cypher = AsyncMock()
        mock_cypher.query = AsyncMock(return_value=[cypher_result])
        mock_embedding = AsyncMock()
        mock_embedding.search = AsyncMock(return_value=[embedding_result])

        service = GraphRetrievalService(
            text_to_cypher=mock_cypher,
            graph_embedding_search=mock_embedding,
        )
        results = await service.search("安心货币A属于什么风险等级", top_k=5)

        assert len(results) == 2
        assert results[0].score >= results[1].score  # RRF sorted

    @pytest.mark.asyncio
    async def test_returns_cypher_only_when_embedding_fails(self):
        cypher_result = GraphSearchResult(
            content="test", entities=[], relations=[],
            score=1.0, source_type="text_to_cypher",
        )
        mock_cypher = AsyncMock()
        mock_cypher.query = AsyncMock(return_value=[cypher_result])
        mock_embedding = AsyncMock()
        mock_embedding.search = AsyncMock(side_effect=Exception("fail"))

        service = GraphRetrievalService(
            text_to_cypher=mock_cypher,
            graph_embedding_search=mock_embedding,
        )
        results = await service.search("test")
        assert len(results) == 1
        assert results[0].source_type == "text_to_cypher"

    @pytest.mark.asyncio
    async def test_returns_empty_when_both_fail(self):
        mock_cypher = AsyncMock()
        mock_cypher.query = AsyncMock(side_effect=Exception("fail"))
        mock_embedding = AsyncMock()
        mock_embedding.search = AsyncMock(side_effect=Exception("fail"))

        service = GraphRetrievalService(
            text_to_cypher=mock_cypher,
            graph_embedding_search=mock_embedding,
        )
        results = await service.search("test")
        assert results == []


class TestMultiPathFusion:
    def test_fuses_vector_and_graph_results(self):
        fusion = MultiPathRetrievalFusion(vector_weight=0.5, graph_weight=0.5)

        vector_results = [
            SearchResult(document_id="v1", content="vector doc 1", score=0.9),
            SearchResult(document_id="v2", content="vector doc 2", score=0.8),
        ]
        graph_results = [
            GraphSearchResult(
                content="graph result 1",
                entities=[GraphEntity(id="1", name="稳盈短债C", type="Fund")],
                relations=[],
                score=0.85,
                source_type="text_to_cypher",
            ),
        ]

        fused = fusion.fuse(vector_results, graph_results, top_k=5)
        assert len(fused) == 3
        assert all(isinstance(r, SearchResult) for r in fused)

    def test_respects_top_k(self):
        fusion = MultiPathRetrievalFusion()
        vector_results = [SearchResult(document_id=f"v{i}", content=f"doc {i}", score=0.9) for i in range(10)]
        graph_results = [GraphSearchResult(content=f"g{i}", entities=[], relations=[], score=0.8, source_type="embedding") for i in range(10)]

        fused = fusion.fuse(vector_results, graph_results, top_k=3)
        assert len(fused) == 3

    def test_handles_empty_inputs(self):
        fusion = MultiPathRetrievalFusion()
        assert fusion.fuse([], [], top_k=5) == []

    def test_vector_only(self):
        fusion = MultiPathRetrievalFusion()
        vector_results = [SearchResult(document_id="v1", content="doc", score=0.9)]
        fused = fusion.fuse(vector_results, [], top_k=5)
        assert len(fused) == 1

    def test_graph_only(self):
        fusion = MultiPathRetrievalFusion()
        graph_results = [GraphSearchResult(content="g", entities=[], relations=[], score=0.8, source_type="emb")]
        fused = fusion.fuse([], graph_results, top_k=5)
        assert len(fused) == 1
        assert fused[0].metadata.get("source_type") == "emb"
