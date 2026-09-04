"""
检索服务包。

导出全部检索组件：向量客户端、混合检索、重排序与工厂。
"""
from app.services.retrieval.vector_base import (
    Document,
    SearchResult,
    VectorSearchRequest,
    VectorClient,
    VectorClientError,
)
from app.services.retrieval.qdrant_client import QdrantClient
from app.services.retrieval.hybrid_search import HybridSearchService, KeywordSearch
from app.services.retrieval.reranking import RerankingService, NoOpReranker
from app.services.retrieval.cross_encoder_reranker import CrossEncoderReranker
from app.services.retrieval.chained_reranker import ChainedReranker
from app.services.retrieval.factory import RetrievalFactory

__all__ = [
    # Base classes and models
    "Document",
    "SearchResult",
    "VectorSearchRequest",
    "VectorClient",
    "VectorClientError",
    # Vector client implementations
    "QdrantClient",
    # Hybrid search
    "HybridSearchService",
    "KeywordSearch",
    # Reranking
    "RerankingService",
    "NoOpReranker",
    "CrossEncoderReranker",
    "ChainedReranker",
    # Factory
    "RetrievalFactory",
]
