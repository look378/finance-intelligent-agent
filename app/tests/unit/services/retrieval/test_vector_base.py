"""Tests for vector service base interface and models"""
import pytest
from dataclasses import asdict


class TestDocument:
    """Test Document dataclass"""

    def test_document_creation(self):
        """Test creating a document"""
        # Arrange
        from app.services.retrieval.vector_base import Document

        # Act
        doc = Document(
            id="doc1",
            content="This is a test document",
            metadata={"category": "test", "source": "unit_test"}
        )

        # Assert
        assert doc.id == "doc1"
        assert doc.content == "This is a test document"
        assert doc.metadata == {"category": "test", "source": "unit_test"}
        assert doc.embedding is None

    def test_document_with_embedding(self):
        """Test creating a document with embedding"""
        # Arrange
        from app.services.retrieval.vector_base import Document

        # Act
        doc = Document(
            id="doc2",
            content="Another document",
            embedding=[0.1, 0.2, 0.3, 0.4]
        )

        # Assert
        assert doc.id == "doc2"
        assert doc.embedding == [0.1, 0.2, 0.3, 0.4]
        assert doc.metadata is None

    def test_document_to_dict(self):
        """Test converting document to dictionary"""
        # Arrange
        from app.services.retrieval.vector_base import Document

        doc = Document(
            id="doc3",
            content="Test content",
            metadata={"key": "value"}
        )

        # Act
        result = asdict(doc)

        # Assert
        assert result["id"] == "doc3"
        assert result["content"] == "Test content"
        assert result["metadata"] == {"key": "value"}


class TestSearchResult:
    """Test SearchResult dataclass"""

    def test_search_result_creation(self):
        """Test creating a search result"""
        # Arrange
        from app.services.retrieval.vector_base import SearchResult

        # Act
        result = SearchResult(
            document_id="doc1",
            content="Test content",
            score=0.95,
            metadata={"category": "test"}
        )

        # Assert
        assert result.document_id == "doc1"
        assert result.content == "Test content"
        assert result.score == 0.95
        assert result.metadata == {"category": "test"}

    def test_search_result_without_metadata(self):
        """Test creating search result without metadata"""
        # Arrange
        from app.services.retrieval.vector_base import SearchResult

        # Act
        result = SearchResult(
            document_id="doc2",
            content="Another content",
            score=0.87
        )

        # Assert
        assert result.document_id == "doc2"
        assert result.metadata is None

    def test_search_result_comparison(self):
        """Test that search results can be compared by score"""
        # Arrange
        from app.services.retrieval.vector_base import SearchResult

        result1 = SearchResult(
            document_id="doc1",
            content="Content 1",
            score=0.95
        )
        result2 = SearchResult(
            document_id="doc2",
            content="Content 2",
            score=0.87
        )

        # Assert
        assert result1.score > result2.score


class TestVectorClient:
    """Test VectorClient abstract class"""

    def test_vector_client_cannot_be_instantiated(self):
        """Test that VectorClient cannot be instantiated directly"""
        # Arrange
        from app.services.retrieval.vector_base import VectorClient

        # Act & Assert
        with pytest.raises(TypeError):
            VectorClient()  # type: ignore

    def test_vector_client_has_abstract_methods(self):
        """Test that VectorClient defines abstract methods"""
        # Arrange
        from app.services.retrieval.vector_base import VectorClient
        import inspect

        # Act
        abstract_methods = inspect.getmembers(VectorClient, predicate=inspect.ismethod)

        # Assert - Check that required abstract methods exist
        assert hasattr(VectorClient, "add_documents")
        assert hasattr(VectorClient, "search")
        assert hasattr(VectorClient, "delete")
        assert hasattr(VectorClient, "update")


class TestVectorSearchRequest:
    """Test VectorSearchRequest dataclass"""

    def test_search_request_creation(self):
        """Test creating a search request"""
        # Arrange
        from app.services.retrieval.vector_base import VectorSearchRequest

        # Act
        request = VectorSearchRequest(
            query="test query",
            top_k=10,
            filters={"category": "tech"}
        )

        # Assert
        assert request.query == "test query"
        assert request.top_k == 10
        assert request.filters == {"category": "tech"}

    def test_search_request_defaults(self):
        """Test search request with default values"""
        # Arrange
        from app.services.retrieval.vector_base import VectorSearchRequest

        # Act
        request = VectorSearchRequest(query="test")

        # Assert
        assert request.query == "test"
        assert request.top_k == 5  # Default
        assert request.filters is None

    def test_search_request_validation(self):
        """Test search request validation"""
        # Arrange
        from app.services.retrieval.vector_base import VectorSearchRequest

        # Act & Assert - Valid requests
        request1 = VectorSearchRequest(query="test", top_k=1)
        assert request1.top_k == 1

        request2 = VectorSearchRequest(query="test", top_k=100)
        assert request2.top_k == 100
