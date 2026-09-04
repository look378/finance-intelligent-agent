"""Tests for retrieval service factory"""
import pytest
from unittest.mock import Mock, patch


class TestRetrievalFactory:
    """Test retrieval service factory"""

    def test_create_qdrant_client(self):
        """Test creating Qdrant client"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory

        # Act
        client = RetrievalFactory.create_vector_client(
            client_type="qdrant",
            url="http://localhost:6333",
            collection_name="test_collection"
        )

        # Assert
        assert client.collection_name == "test_collection"

    def test_create_qdrant_client_with_api_key(self):
        """Test creating Qdrant client with API key"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory

        # Act
        client = RetrievalFactory.create_vector_client(
            client_type="qdrant",
            url="http://localhost:6333",
            collection_name="test_collection",
            api_key="test_api_key"
        )

        # Assert
        assert client.api_key == "test_api_key"

    def test_create_invalid_client_type(self):
        """Test creating invalid client type raises error"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory
        from app.core.exceptions import ValidationError

        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            RetrievalFactory.create_vector_client(
                client_type="invalid_type",
                url="http://localhost:6333",
                collection_name="test_collection"
            )
        assert "client" in str(exc_info.value).lower()

    def test_create_hybrid_search_service(self):
        """Test creating hybrid search service"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory

        mock_vector_client = Mock()
        mock_keyword_search = Mock()

        # Act
        service = RetrievalFactory.create_hybrid_search(
            vector_client=mock_vector_client,
            keyword_search=mock_keyword_search,
            vector_weight=0.7
        )

        # Assert（浮点精度用 approx）
        assert service.vector_weight == 0.7
        assert service.keyword_weight == pytest.approx(0.3)

    def test_create_hybrid_search_default_weights(self):
        """Test creating hybrid search with default weights"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory

        mock_vector_client = Mock()
        mock_keyword_search = Mock()

        # Act
        service = RetrievalFactory.create_hybrid_search(
            vector_client=mock_vector_client,
            keyword_search=mock_keyword_search
        )

        # Assert
        assert service.vector_weight == 0.5
        assert service.keyword_weight == 0.5

    def test_create_reranking_service(self):
        """Test creating reranking service"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory

        mock_llm = Mock()

        # Act
        reranker = RetrievalFactory.create_reranker(
            llm_service=mock_llm,
            top_n=10
        )

        # Assert
        assert reranker.top_n == 10

    def test_create_reranking_service_default_top_n(self):
        """Test creating reranker with default top_n"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory

        mock_llm = Mock()

        # Act
        reranker = RetrievalFactory.create_reranker(
            llm_service=mock_llm
        )

        # Assert
        assert reranker.top_n == 5

    def test_create_noop_reranker(self):
        """Test creating no-op reranker"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory

        # Act
        reranker = RetrievalFactory.create_reranker(
            reranker_type="noop"
        )

        # Assert
        from app.services.retrieval.reranking import NoOpReranker
        assert isinstance(reranker, NoOpReranker)

    def test_create_retrieval_pipeline(self):
        """Test creating complete retrieval pipeline"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory

        mock_vector_client = Mock()
        mock_llm = Mock()

        # Act
        pipeline = RetrievalFactory.create_pipeline(
            vector_client=mock_vector_client,
            llm_service=mock_llm,
            use_reranking=True,
        )

        # Assert
        assert pipeline["vector_client"] == mock_vector_client
        assert pipeline["reranker"] is not None

    def test_create_retrieval_pipeline_without_reranking(self):
        """Test creating pipeline without reranking"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory

        mock_vector_client = Mock()
        mock_llm = Mock()

        # Act
        pipeline = RetrievalFactory.create_pipeline(
            vector_client=mock_vector_client,
            llm_service=mock_llm,
            use_reranking=False
        )

        # Assert
        from app.services.retrieval.reranking import NoOpReranker
        assert isinstance(pipeline["reranker"], NoOpReranker)

    def test_factory_config_validation(self):
        """Test factory configuration validation"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory
        from app.core.exceptions import ValidationError

        # Act & Assert - Invalid vector weight
        with pytest.raises(ValidationError):
            RetrievalFactory.create_hybrid_search(
                vector_client=Mock(),
                keyword_search=Mock(),
                vector_weight=1.5  # Invalid
            )

    def test_create_with_custom_reranker_config(self):
        """Test creating reranker with custom configuration"""
        # Arrange
        from app.services.retrieval.factory import RetrievalFactory

        mock_llm = Mock()

        # Act
        reranker = RetrievalFactory.create_reranker(
            llm_service=mock_llm,
            reranker_type="llm",
            top_n=15
        )

        # Assert
        assert reranker.top_n == 15

    def test_create_cross_encoder_reranker(self):
        """Test creating cross-encoder reranker"""
        from app.services.retrieval.factory import RetrievalFactory
        from app.services.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = RetrievalFactory.create_reranker(
            reranker_type="cross_encoder",
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            device="cpu",
            top_n=10,
        )

        assert isinstance(reranker, CrossEncoderReranker)
        assert reranker.top_n == 10
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_create_cross_encoder_reranker_default_model(self):
        """Test creating cross-encoder reranker with default model"""
        from app.services.retrieval.factory import RetrievalFactory
        from app.services.retrieval.cross_encoder_reranker import CrossEncoderReranker

        reranker = RetrievalFactory.create_reranker(reranker_type="cross_encoder")

        assert isinstance(reranker, CrossEncoderReranker)
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_create_invalid_reranker_type(self):
        """Test creating invalid reranker type raises error"""
        from app.services.retrieval.factory import RetrievalFactory
        from app.core.exceptions import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            RetrievalFactory.create_reranker(reranker_type="invalid")

        assert "cross_encoder" in str(exc_info.value)

    def test_create_reranker_from_settings_disabled(self):
        """Test create_reranker_from_settings returns NoOpReranker when disabled"""
        from app.services.retrieval.factory import RetrievalFactory
        from app.services.retrieval.reranking import NoOpReranker

        mock_settings = Mock()
        mock_settings.RERANKER_ENABLED = False

        with patch("app.config.settings.get_settings", return_value=mock_settings):
            reranker = RetrievalFactory.create_reranker_from_settings()

        assert isinstance(reranker, NoOpReranker)

    def test_create_reranker_from_settings_cross_encoder(self):
        """Test create_reranker_from_settings with cross_encoder type"""
        from app.services.retrieval.factory import RetrievalFactory
        from app.services.retrieval.cross_encoder_reranker import CrossEncoderReranker

        mock_settings = Mock()
        mock_settings.RERANKER_ENABLED = True
        mock_settings.RERANKER_TYPE = "cross_encoder"
        mock_settings.RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        mock_settings.RERANKER_DEVICE = "cpu"
        mock_settings.RERANKER_TOP_N = 5
        mock_settings.RERANKER_LLM_SECOND_STAGE = False

        with patch("app.config.settings.get_settings", return_value=mock_settings):
            reranker = RetrievalFactory.create_reranker_from_settings()

        assert isinstance(reranker, CrossEncoderReranker)

    def test_create_reranker_from_settings_chained_with_llm(self):
        """Test create_reranker_from_settings with chained type (cross_encoder + LLM)"""
        from app.services.retrieval.factory import RetrievalFactory
        from app.services.retrieval.chained_reranker import ChainedReranker

        mock_settings = Mock()
        mock_settings.RERANKER_ENABLED = True
        mock_settings.RERANKER_TYPE = "cross_encoder"
        mock_settings.RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        mock_settings.RERANKER_DEVICE = "cpu"
        mock_settings.RERANKER_TOP_N = 5
        mock_settings.RERANKER_LLM_SECOND_STAGE = True
        mock_settings.RERANKER_LLM_TOP_N = 3

        mock_llm = Mock()

        with patch("app.config.settings.get_settings", return_value=mock_settings):
            reranker = RetrievalFactory.create_reranker_from_settings(llm_service=mock_llm)

        assert isinstance(reranker, ChainedReranker)
        assert reranker.second_stage is not None
        assert reranker.second_stage.top_n == 3
