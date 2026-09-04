"""Tests for cross-encoder reranking service."""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from app.services.retrieval.vector_base import SearchResult, VectorSearchRequest


class TestCrossEncoderReranker:
    """Test cross-encoder reranker."""

    def test_initialization(self):
        from app.services.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu",
            top_n=5,
        )

        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker.device == "cpu"
        assert reranker.top_n == 5
        assert reranker._model is None

    def test_initialization_defaults(self):
        from app.services.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker.device == "cpu"
        assert reranker.top_n == 5

    @pytest.mark.asyncio
    async def test_rerank_empty_results(self):
        from app.services.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        request = VectorSearchRequest(query="test")

        result = await reranker.rerank([], request)

        assert result == []

    @pytest.mark.asyncio
    @patch("app.services.retrieval.cross_encoder_reranker.CrossEncoderReranker._load_model")
    async def test_rerank_orders_by_score(self, mock_load):
        from app.services.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(top_n=3)

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.9, 0.5])
        reranker._model = mock_model

        results = [
            SearchResult(document_id="doc1", content="low relevance", score=0.9),
            SearchResult(document_id="doc2", content="high relevance", score=0.3),
            SearchResult(document_id="doc3", content="medium relevance", score=0.6),
        ]
        request = VectorSearchRequest(query="test query")

        reranked = await reranker.rerank(results, request)

        assert len(reranked) == 3
        assert reranked[0].document_id == "doc2"
        assert reranked[1].document_id == "doc3"
        assert reranked[2].document_id == "doc1"
        assert reranked[0].score == pytest.approx(0.9)

    @pytest.mark.asyncio
    @patch("app.services.retrieval.cross_encoder_reranker.CrossEncoderReranker._load_model")
    async def test_rerank_respects_top_n(self, mock_load):
        from app.services.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker(top_n=2)

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.1, 0.9, 0.5, 0.8, 0.3])
        reranker._model = mock_model

        results = [
            SearchResult(document_id=f"doc{i}", content=f"content {i}", score=0.5)
            for i in range(5)
        ]
        request = VectorSearchRequest(query="test")

        reranked = await reranker.rerank(results, request)

        assert len(reranked) == 2
        assert reranked[0].document_id == "doc1"
        assert reranked[1].document_id == "doc3"

    @pytest.mark.asyncio
    @patch("app.services.retrieval.cross_encoder_reranker.CrossEncoderReranker._load_model")
    async def test_rerank_preserves_metadata(self, mock_load):
        from app.services.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.8])
        reranker._model = mock_model

        results = [
            SearchResult(
                document_id="doc1",
                content="content",
                score=0.5,
                metadata={"category": "tech"},
            ),
        ]
        request = VectorSearchRequest(query="test")

        reranked = await reranker.rerank(results, request)

        assert len(reranked) == 1
        assert reranked[0].metadata["category"] == "tech"
        assert reranked[0].metadata["original_score"] == 0.5
        assert reranked[0].metadata["reranker"] == "cross_encoder"

    @pytest.mark.asyncio
    @patch("app.services.retrieval.cross_encoder_reranker.CrossEncoderReranker._load_model")
    async def test_rerank_lazy_loads_model(self, mock_load):
        from app.services.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        assert reranker._model is None

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.5])
        reranker._model = mock_model

        results = [SearchResult(document_id="doc1", content="content", score=0.5)]
        request = VectorSearchRequest(query="test")

        await reranker.rerank(results, request)

        reranker._model.predict.assert_called_once()

    @pytest.mark.asyncio
    @patch("app.services.retrieval.cross_encoder_reranker.CrossEncoderReranker._load_model")
    async def test_rerank_handles_predict_error(self, mock_load):
        from app.services.retrieval.cross_encoder_reranker import CrossEncoderReranker
        from app.core.exceptions import ExternalServiceError

        reranker = CrossEncoderReranker()

        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("model error")
        reranker._model = mock_model

        results = [SearchResult(document_id="doc1", content="content", score=0.5)]
        request = VectorSearchRequest(query="test")

        with pytest.raises(ExternalServiceError):
            await reranker.rerank(results, request)

    @pytest.mark.asyncio
    async def test_load_model_success(self):
        from app.services.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()

        mock_ce = MagicMock()
        with patch(
            "sentence_transformers.CrossEncoder",
            return_value=mock_ce,
        ) as mock_cls:
            await reranker._load_model()
            mock_cls.assert_called_once_with(
                "cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"
            )
            assert reranker._model is mock_ce

    @pytest.mark.asyncio
    async def test_load_model_idempotent(self):
        from app.services.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = CrossEncoderReranker()
        reranker._model = MagicMock()

        with patch(
            "sentence_transformers.CrossEncoder"
        ) as mock_cls:
            await reranker._load_model()
            mock_cls.assert_not_called()
