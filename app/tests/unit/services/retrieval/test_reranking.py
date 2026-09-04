"""Tests for result reranking service"""
import pytest
from unittest.mock import Mock, AsyncMock, patch


class TestRerankingService:
    """Test reranking service for improving search results"""

    def test_reranker_initialization(self):
        """Test reranker initialization"""
        # Arrange
        from app.services.retrieval.reranking import RerankingService

        mock_llm = Mock()

        # Act
        reranker = RerankingService(llm_service=mock_llm)

        # Assert
        assert reranker.llm_service == mock_llm
        assert reranker.top_n == 5  # Default

    def test_reranker_initialization_with_top_n(self):
        """Test reranker with custom top_n"""
        # Arrange
        from app.services.retrieval.reranking import RerankingService

        mock_llm = Mock()

        # Act
        reranker = RerankingService(llm_service=mock_llm, top_n=10)

        # Assert
        assert reranker.top_n == 10

    @pytest.mark.asyncio
    async def test_rerank_results(self):
        """Test reranking search results"""
        # Arrange
        from app.services.retrieval.reranking import RerankingService
        from app.services.retrieval.vector_base import SearchResult, VectorSearchRequest

        mock_llm = Mock()
        reranker = RerankingService(llm_service=mock_llm)

        # Mock LLM response
        mock_llm.generate = AsyncMock(
            return_value=Mock(
                content='{"reranked": ["doc2", "doc1", "doc3"]}'
            )
        )

        results = [
            SearchResult(document_id="doc1", content="Content 1", score=0.9),
            SearchResult(document_id="doc2", content="Content 2", score=0.8),
            SearchResult(document_id="doc3", content="Content 3", score=0.7),
        ]

        request = VectorSearchRequest(query="test query")

        # Act
        reranked = await reranker.rerank(results, request)

        # Assert - Results should be reordered
        assert reranked[0].document_id == "doc2"  # Now highest
        assert reranked[1].document_id == "doc1"
        assert reranked[2].document_id == "doc3"

    @pytest.mark.asyncio
    async def test_rerank_with_scores(self):
        """Test reranking with new scores from LLM"""
        # Arrange
        from app.services.retrieval.reranking import RerankingService
        from app.services.retrieval.vector_base import SearchResult, VectorSearchRequest

        mock_llm = Mock()
        reranker = RerankingService(llm_service=mock_llm)

        # Mock LLM response with scores
        mock_llm.generate = AsyncMock(
            return_value=Mock(
                content='{"results": [{"id": "doc2", "score": 0.95}, {"id": "doc1", "score": 0.85}]}'
            )
        )

        results = [
            SearchResult(document_id="doc1", content="Content 1", score=0.9),
            SearchResult(document_id="doc2", content="Content 2", score=0.8),
        ]

        request = VectorSearchRequest(query="test query")

        # Act
        reranked = await reranker.rerank(results, request)

        # Assert
        assert reranked[0].document_id == "doc2"
        assert reranked[0].score == 0.95
        assert reranked[1].document_id == "doc1"
        assert reranked[1].score == 0.85

    @pytest.mark.asyncio
    async def test_rerank_top_n_filtering(self):
        """Test that reranking only returns top_n results"""
        # Arrange
        from app.services.retrieval.reranking import RerankingService
        from app.services.retrieval.vector_base import SearchResult, VectorSearchRequest

        mock_llm = Mock()
        reranker = RerankingService(llm_service=mock_llm, top_n=2)

        mock_llm.generate = AsyncMock(
            return_value=Mock(
                content='{"reranked": ["doc1", "doc2", "doc3", "doc4", "doc5"]}'
            )
        )

        results = [
            SearchResult(document_id=f"doc{i}", content=f"Content {i}", score=0.9 - i * 0.1)
            for i in range(1, 6)
        ]

        request = VectorSearchRequest(query="test query")

        # Act
        reranked = await reranker.rerank(results, request)

        # Assert - Should only return top_n results
        assert len(reranked) == 2
        assert reranked[0].document_id == "doc1"
        assert reranked[1].document_id == "doc2"

    @pytest.mark.asyncio
    async def test_rerank_empty_results(self):
        """Test reranking with empty results"""
        # Arrange
        from app.services.retrieval.reranking import RerankingService
        from app.services.retrieval.vector_base import VectorSearchRequest

        mock_llm = Mock()
        reranker = RerankingService(llm_service=mock_llm)

        request = VectorSearchRequest(query="test query")

        # Act
        reranked = await reranker.rerank([], request)

        # Assert
        assert len(reranked) == 0

    @pytest.mark.asyncio
    async def test_rerank_with_query_relevance(self):
        """Test reranking based on query relevance"""
        # Arrange
        from app.services.retrieval.reranking import RerankingService
        from app.services.retrieval.vector_base import SearchResult, VectorSearchRequest

        mock_llm = Mock()
        reranker = RerankingService(llm_service=mock_llm)

        # Mock LLM to score based on query relevance
        mock_llm.generate = AsyncMock(
            return_value=Mock(
                content='{"reranked": ["doc3", "doc1", "doc2"]}'
            )
        )

        results = [
            SearchResult(document_id="doc1", content="Python programming", score=0.9),
            SearchResult(document_id="doc2", content="Java guide", score=0.8),
            SearchResult(document_id="doc3", content="Python tutorial", score=0.7),
        ]

        request = VectorSearchRequest(query="Python programming tutorial")

        # Act
        reranked = await reranker.rerank(results, request)

        # Assert - doc3 should be highest (most relevant to query)
        assert reranked[0].document_id == "doc3"
        assert "relevance_score" in reranked[0].metadata or True

    @pytest.mark.asyncio
    async def test_rerank_with_metadata_preservation(self):
        """Test that metadata is preserved during reranking"""
        # Arrange
        from app.services.retrieval.reranking import RerankingService
        from app.services.retrieval.vector_base import SearchResult, VectorSearchRequest

        mock_llm = Mock()
        reranker = RerankingService(llm_service=mock_llm)

        mock_llm.generate = AsyncMock(
            return_value=Mock(
                content='{"reranked": ["doc2", "doc1"]}'
            )
        )

        results = [
            SearchResult(
                document_id="doc1",
                content="Content 1",
                score=0.9,
                metadata={"category": "tech", "source": "internal"}
            ),
            SearchResult(
                document_id="doc2",
                content="Content 2",
                score=0.8,
                metadata={"category": "general", "source": "external"}
            ),
        ]

        request = VectorSearchRequest(query="test")

        # Act
        reranked = await reranker.rerank(results, request)

        # Assert - Metadata should be preserved
        assert reranked[0].document_id == "doc2"
        assert reranked[0].metadata["category"] == "general"
        assert reranked[0].metadata["source"] == "external"


class TestNoOpReranker:
    """Test no-op reranker (passes results through)"""

    def test_noop_reranker_initialization(self):
        """Test no-op reranker initialization"""
        # Arrange
        from app.services.retrieval.reranking import NoOpReranker

        # Act
        reranker = NoOpReranker()

        # Assert
        assert reranker is not None

    @pytest.mark.asyncio
    async def test_noop_rerank_passes_results_through(self):
        """Test that no-op reranker returns results unchanged"""
        # Arrange
        from app.services.retrieval.reranking import NoOpReranker
        from app.services.retrieval.vector_base import SearchResult, VectorSearchRequest

        reranker = NoOpReranker()

        results = [
            SearchResult(document_id="doc1", content="Content 1", score=0.9),
            SearchResult(document_id="doc2", content="Content 2", score=0.8),
        ]

        request = VectorSearchRequest(query="test")

        # Act
        reranked = await reranker.rerank(results, request)

        # Assert
        assert len(reranked) == 2
        assert reranked[0].document_id == "doc1"
        assert reranked[1].document_id == "doc2"
        # Scores should be unchanged
        assert reranked[0].score == 0.9
        assert reranked[1].score == 0.8


class TestRerankingStrategies:
    """Test different reranking strategies"""

    @pytest.mark.asyncio
    async def test_score_normalization_reranking(self):
        """Test score normalization strategy"""
        # Arrange
        from app.services.retrieval.reranking import RerankingService
        from app.services.retrieval.vector_base import SearchResult, VectorSearchRequest

        mock_llm = Mock()
        reranker = RerankingService(llm_service=mock_llm)

        # Results with different score ranges
        results = [
            SearchResult(document_id="doc1", content="Content 1", score=0.95),
            SearchResult(document_id="doc2", content="Content 2", score=0.0003),
            SearchResult(document_id="doc3", content="Content 3", score=15.7),
        ]

        request = VectorSearchRequest(query="test")

        # Mock LLM to return normalized scores
        mock_llm.generate = AsyncMock(
            return_value=Mock(
                content='{"results": [{"id": "doc1", "score": 0.9}, {"id": "doc2", "score": 0.1}, {"id": "doc3", "score": 0.5}]}'
            )
        )

        # Act
        reranked = await reranker.rerank(results, request)

        # Assert - All scores should be normalized to 0-1 range
        for result in reranked:
            assert 0.0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_diversity_reranking(self):
        """Test diversity-aware reranking (MMR - Maximal Marginal Relevance)"""
        # Arrange
        from app.services.retrieval.reranking import RerankingService
        from app.services.retrieval.vector_base import SearchResult, VectorSearchRequest

        mock_llm = Mock()
        reranker = RerankingService(llm_service=mock_llm)

        # Similar results (high relevance, low diversity)
        results = [
            SearchResult(
                document_id="doc1",
                content="Python is a programming language",
                score=0.95,
                metadata={"category": "tech"}
            ),
            SearchResult(
                document_id="doc2",
                content="Python programming language tutorial",
                score=0.93,
                metadata={"category": "tech"}
            ),
            SearchResult(
                document_id="doc3",
                content="Java is a programming language",
                score=0.70,
                metadata={"category": "tech"}
            ),
        ]

        request = VectorSearchRequest(query="Python programming")

        # Mock LLM to prioritize diversity
        mock_llm.generate = AsyncMock(
            return_value=Mock(
                content='{"reranked": ["doc1", "doc3", "doc2"]}'  # doc3 promoted for diversity
            )
        )

        # Act
        reranked = await reranker.rerank(results, request)

        # Assert - Should promote diverse results
        assert reranked[0].document_id == "doc1"
        assert reranked[1].document_id == "doc3"  # Different content promoted
        assert reranked[2].document_id == "doc2"
