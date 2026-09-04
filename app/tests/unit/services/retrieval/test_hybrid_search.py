"""Tests for hybrid search service"""
import pytest
from unittest.mock import Mock, AsyncMock, patch


class TestHybridSearchService:
    """Test hybrid search service combining vector and keyword search"""

    def test_service_initialization(self):
        """Test hybrid search service initialization"""
        # Arrange
        from app.services.retrieval.hybrid_search import HybridSearchService

        mock_vector_client = Mock()
        mock_keyword_search = Mock()

        # Act
        service = HybridSearchService(
            vector_client=mock_vector_client,
            keyword_search=mock_keyword_search,
            vector_weight=0.7
        )

        # Assert
        assert service.vector_weight == 0.7
        assert service.keyword_weight == 0.3  # 1.0 - 0.7

    def test_service_initialization_default_weights(self):
        """Test service initialization with default weights"""
        # Arrange
        from app.services.retrieval.hybrid_search import HybridSearchService

        mock_vector_client = Mock()
        mock_keyword_search = Mock()

        # Act
        service = HybridSearchService(
            vector_client=mock_vector_client,
            keyword_search=mock_keyword_search
        )

        # Assert
        assert service.vector_weight == 0.5  # Default
        assert service.keyword_weight == 0.5  # Default

    def test_weight_validation(self):
        """Test that weights must sum to 1.0"""
        # Arrange
        from app.services.retrieval.hybrid_search import HybridSearchService
        from app.core.exceptions import ValidationError

        mock_vector_client = Mock()
        mock_keyword_search = Mock()

        # Act & Assert
        with pytest.raises(ValidationError):
            HybridSearchService(
                vector_client=mock_vector_client,
                keyword_search=mock_keyword_search,
                vector_weight=0.8  # Should fail (0.8 + 0.5 != 1.0)
            )

    @pytest.mark.asyncio
    async def test_hybrid_search(self):
        """Test hybrid search combining vector and keyword results"""
        # Arrange
        from app.services.retrieval.hybrid_search import HybridSearchService
        from app.services.retrieval.vector_base import VectorSearchRequest, SearchResult

        mock_vector_client = Mock()
        mock_keyword_search = Mock()

        # Mock vector search results
        vector_results = [
            SearchResult(document_id="doc1", content="Content 1", score=0.9, metadata={"source": "vector"}),
            SearchResult(document_id="doc2", content="Content 2", score=0.8, metadata={"source": "vector"}),
        ]
        mock_vector_client.search = AsyncMock(return_value=vector_results)

        # Mock keyword search results
        keyword_results = [
            SearchResult(document_id="doc1", content="Content 1", score=0.7, metadata={"source": "keyword"}),
            SearchResult(document_id="doc3", content="Content 3", score=0.6, metadata={"source": "keyword"}),
        ]
        mock_keyword_search.search = AsyncMock(return_value=keyword_results)

        service = HybridSearchService(
            vector_client=mock_vector_client,
            keyword_search=mock_keyword_search,
            vector_weight=0.6
        )

        request = VectorSearchRequest(query="test query", top_k=5)

        # Act
        results = await service.search(request)

        # Assert - Results should be combined and reranked
        assert len(results) == 3  # doc1, doc2, doc3

        # doc1 should have highest score (appears in both)
        doc1_result = next(r for r in results if r.document_id == "doc1")
        assert doc1_result.score > 0.7  # Combined score

        # Verify metadata shows hybrid scoring
        assert doc1_result.metadata is not None
        assert "vector_score" in doc1_result.metadata
        assert "keyword_score" in doc1_result.metadata

    @pytest.mark.asyncio
    async def test_hybrid_search_vector_only(self):
        """Test hybrid search when keyword search fails"""
        # Arrange
        from app.services.retrieval.hybrid_search import HybridSearchService
        from app.services.retrieval.vector_base import VectorSearchRequest, SearchResult

        mock_vector_client = Mock()
        mock_keyword_search = Mock()

        vector_results = [
            SearchResult(document_id="doc1", content="Content 1", score=0.9),
        ]
        mock_vector_client.search = AsyncMock(return_value=vector_results)
        mock_keyword_search.search = AsyncMock(side_effect=Exception("Keyword search failed"))

        service = HybridSearchService(
            vector_client=mock_vector_client,
            keyword_search=mock_keyword_search
        )

        request = VectorSearchRequest(query="test query")

        # Act
        results = await service.search(request)

        # Assert - Should fall back to vector only
        assert len(results) == 1
        assert results[0].document_id == "doc1"

    @pytest.mark.asyncio
    async def test_hybrid_search_keyword_only(self):
        """Test hybrid search when vector search fails"""
        # Arrange
        from app.services.retrieval.hybrid_search import HybridSearchService
        from app.services.retrieval.vector_base import VectorSearchRequest, SearchResult

        mock_vector_client = Mock()
        mock_keyword_search = Mock()

        mock_vector_client.search = AsyncMock(side_effect=Exception("Vector search failed"))
        keyword_results = [
            SearchResult(document_id="doc2", content="Content 2", score=0.8),
        ]
        mock_keyword_search.search = AsyncMock(return_value=keyword_results)

        service = HybridSearchService(
            vector_client=mock_vector_client,
            keyword_search=mock_keyword_search
        )

        request = VectorSearchRequest(query="test query")

        # Act
        results = await service.search(request)

        # Assert - Should fall back to keyword only
        assert len(results) == 1
        assert results[0].document_id == "doc2"

    @pytest.mark.asyncio
    async def test_hybrid_search_top_k_limit(self):
        """Test that hybrid search respects top_k limit"""
        # Arrange
        from app.services.retrieval.hybrid_search import HybridSearchService
        from app.services.retrieval.vector_base import VectorSearchRequest, SearchResult

        mock_vector_client = Mock()
        mock_keyword_search = Mock()

        # Mock many results
        vector_results = [
            SearchResult(document_id=f"doc{i}", content=f"Content {i}", score=0.9 - i * 0.1)
            for i in range(10)
        ]
        mock_vector_client.search = AsyncMock(return_value=vector_results)
        mock_keyword_search.search = AsyncMock(return_value=[])

        service = HybridSearchService(
            vector_client=mock_vector_client,
            keyword_search=mock_keyword_search
        )

        request = VectorSearchRequest(query="test query", top_k=5)

        # Act
        results = await service.search(request)

        # Assert
        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_score_normalization(self):
        """Test that scores are properly normalized"""
        # Arrange
        from app.services.retrieval.hybrid_search import HybridSearchService
        from app.services.retrieval.vector_base import VectorSearchRequest, SearchResult

        mock_vector_client = Mock()
        mock_keyword_search = Mock()

        vector_results = [
            SearchResult(document_id="doc1", content="Content 1", score=1.0),
        ]
        mock_vector_client.search = AsyncMock(return_value=vector_results)

        keyword_results = [
            SearchResult(document_id="doc1", content="Content 1", score=0.5),
        ]
        mock_keyword_search.search = AsyncMock(return_value=keyword_results)

        service = HybridSearchService(
            vector_client=mock_vector_client,
            keyword_search=mock_keyword_search,
            vector_weight=0.5
        )

        request = VectorSearchRequest(query="test query")

        # Act
        results = await service.search(request)

        # Assert - Combined score should be normalized
        assert len(results) == 1
        assert 0.0 <= results[0].score <= 1.0

    def test_reciprocal_rank_fusion(self):
        """Test Reciprocal Rank Fusion (RRF) algorithm"""
        # Arrange
        from app.services.retrieval.hybrid_search import HybridSearchService
        from app.services.retrieval.vector_base import SearchResult

        service = HybridSearchService(
            vector_client=Mock(),
            keyword_search=Mock()
        )

        # Create two ranked lists
        vector_results = [
            SearchResult(document_id="doc1", content="Content 1", score=0.9),
            SearchResult(document_id="doc2", content="Content 2", score=0.8),
            SearchResult(document_id="doc3", content="Content 3", score=0.7),
        ]

        keyword_results = [
            SearchResult(document_id="doc2", content="Content 2", score=0.95),
            SearchResult(document_id="doc1", content="Content 1", score=0.85),
            SearchResult(document_id="doc4", content="Content 4", score=0.75),
        ]

        # Act
        fused = service._reciprocal_rank_fusion(vector_results, keyword_results, k=60)

        # Assert - RRF should combine rankings
        # doc1: ranks 1 (vector) + 2 (keyword) = 1/60 + 1/120 ≈ 0.025
        # doc2: ranks 2 (vector) + 1 (keyword) = 1/120 + 1/60 ≈ 0.025
        # doc3: ranks 3 (vector) = 1/180 ≈ 0.0056
        # doc4: ranks 3 (keyword) = 1/180 ≈ 0.0056

        assert len(fused) == 4  # All unique documents

        # doc1 and doc2 should have highest scores (appear in both lists)
        doc1 = next(r for r in fused if r.document_id == "doc1")
        doc2 = next(r for r in fused if r.document_id == "doc2")
        doc3 = next(r for r in fused if r.document_id == "doc3")
        doc4 = next(r for r in fused if r.document_id == "doc4")

        assert doc1.score > doc3.score
        assert doc2.score > doc4.score


class TestKeywordSearch:
    """Test keyword search implementation"""

    def test_keyword_search_initialization(self):
        """Test keyword search initialization"""
        # Arrange
        from app.services.retrieval.hybrid_search import KeywordSearch

        # Act
        search = KeywordSearch()

        # Assert
        assert search is not None

    @pytest.mark.asyncio
    async def test_keyword_search_basic(self):
        """Test basic keyword search"""
        # Arrange
        from app.services.retrieval.hybrid_search import KeywordSearch
        from app.services.retrieval.vector_base import VectorSearchRequest

        search = KeywordSearch()

        # Add some documents
        await search.add_documents([
            {"id": "doc1", "content": "Python programming tutorial"},
            {"id": "doc2", "content": "Java programming guide"},
            {"id": "doc3", "content": "Python data science"},
        ])

        request = VectorSearchRequest(query="Python programming")

        # Act
        results = await search.search(request)

        # Assert
        assert len(results) > 0
        # doc1 should rank highest (matches both "Python" and "programming")
        assert results[0].document_id == "doc1"

    @pytest.mark.asyncio
    async def test_keyword_search_empty_query(self):
        """Test keyword search with empty query"""
        # Arrange
        from app.services.retrieval.hybrid_search import KeywordSearch
        from app.services.retrieval.vector_base import VectorSearchRequest

        search = KeywordSearch()
        await search.add_documents([
            {"id": "doc1", "content": "Test content"}
        ])

        request = VectorSearchRequest(query="")

        # Act
        results = await search.search(request)

        # Assert - Should return all results with same score
        assert len(results) == 1
        assert results[0].score == 0.0  # Empty query gets zero score
