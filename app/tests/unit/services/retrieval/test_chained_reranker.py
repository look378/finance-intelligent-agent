"""Tests for chained reranker (two-stage orchestration)."""
import pytest
from unittest.mock import AsyncMock, Mock

from app.services.retrieval.vector_base import SearchResult, VectorSearchRequest


class TestChainedReranker:
    """Test two-stage chained reranker."""

    @pytest.mark.asyncio
    async def test_single_stage_no_second(self):
        from app.services.retrieval.chained_reranker import ChainedReranker

        mock_first = Mock()
        mock_first.rerank = AsyncMock(return_value=[
            SearchResult(document_id="doc1", content="a", score=0.9),
        ])

        chained = ChainedReranker(first_stage=mock_first, second_stage=None)
        request = VectorSearchRequest(query="test")
        results = [SearchResult(document_id="doc1", content="a", score=0.5)]

        reranked = await chained.rerank(results, request)

        mock_first.rerank.assert_called_once_with(results, request)
        assert len(reranked) == 1
        assert reranked[0].document_id == "doc1"

    @pytest.mark.asyncio
    async def test_two_stage_calls_both(self):
        from app.services.retrieval.chained_reranker import ChainedReranker

        first_output = [
            SearchResult(document_id="doc1", content="a", score=0.9),
            SearchResult(document_id="doc2", content="b", score=0.8),
        ]
        second_output = [
            SearchResult(document_id="doc1", content="a", score=0.95),
        ]

        mock_first = Mock()
        mock_first.rerank = AsyncMock(return_value=first_output)
        mock_second = Mock()
        mock_second.rerank = AsyncMock(return_value=second_output)

        chained = ChainedReranker(first_stage=mock_first, second_stage=mock_second)
        request = VectorSearchRequest(query="test")
        results = [SearchResult(document_id="doc1", content="a", score=0.5)]

        reranked = await chained.rerank(results, request)

        mock_first.rerank.assert_called_once_with(results, request)
        mock_second.rerank.assert_called_once_with(first_output, request)
        assert len(reranked) == 1
        assert reranked[0].document_id == "doc1"
        assert reranked[0].score == 0.95

    @pytest.mark.asyncio
    async def test_empty_results(self):
        from app.services.retrieval.chained_reranker import ChainedReranker

        mock_first = Mock()
        mock_first.rerank = AsyncMock(return_value=[])

        chained = ChainedReranker(first_stage=mock_first)
        request = VectorSearchRequest(query="test")

        reranked = await chained.rerank([], request)

        assert reranked == []

    @pytest.mark.asyncio
    async def test_first_stage_returns_empty_skips_second(self):
        from app.services.retrieval.chained_reranker import ChainedReranker

        mock_first = Mock()
        mock_first.rerank = AsyncMock(return_value=[])
        mock_second = Mock()
        mock_second.rerank = AsyncMock()

        chained = ChainedReranker(first_stage=mock_first, second_stage=mock_second)
        request = VectorSearchRequest(query="test")
        results = [SearchResult(document_id="doc1", content="a", score=0.5)]

        reranked = await chained.rerank(results, request)

        mock_second.rerank.assert_not_called()
        assert reranked == []
