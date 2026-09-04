"""
Factory for creating retrieval service components.

Provides simple interface for creating vector clients, hybrid search,
reranking services, and complete retrieval pipelines.
"""
from typing import Optional

from app.services.retrieval.vector_base import VectorClient
from app.services.retrieval.qdrant_client import QdrantClient
from app.services.retrieval.hybrid_search import HybridSearchService, KeywordSearch
from app.services.retrieval.reranking import RerankingService, NoOpReranker
from app.services.retrieval.cross_encoder_reranker import CrossEncoderReranker
from app.services.retrieval.chained_reranker import ChainedReranker
from app.services.llm.base import LLMServiceBase
from app.core.exceptions import ValidationError


class RetrievalFactory:
    """
    Factory for creating retrieval service components.

    Provides methods for creating vector clients, hybrid search services,
    rerankers, metadata services, and complete retrieval pipelines.
    """

    @staticmethod
    def create_vector_client(
        client_type: str,
        url: str,
        collection_name: str,
        api_key: Optional[str] = None,
        embedding_service=None,
        **kwargs,
    ) -> VectorClient:
        """
        Create a vector database client.

        Args:
            client_type: Type of client ("qdrant")
            url: Vector database URL
            collection_name: Collection/table name
            api_key: Optional API key
            embedding_service: Optional embedding service for generating embeddings
            **kwargs: Additional client-specific parameters

        Returns:
            VectorClient: Configured vector client

        Raises:
            ValidationError: If client_type is invalid
        """
        if client_type == "qdrant":
            return QdrantClient(
                url=url,
                collection_name=collection_name,
                api_key=api_key,
                embedding_service=embedding_service,
                **kwargs,
            )
        else:
            valid_types = ["qdrant"]
            raise ValidationError(
                f"Invalid client_type: {client_type}. "
                f"Must be one of {valid_types}"
            )

    @staticmethod
    def create_hybrid_search(
        vector_client: VectorClient,
        vector_weight: float = 0.5,
        keyword_search: Optional[KeywordSearch] = None,
    ) -> HybridSearchService:
        """
        Create a hybrid search service.

        Args:
            vector_client: Vector search client
            vector_weight: Weight for vector search (0.0-1.0)
            keyword_search: Optional keyword search instance

        Returns:
            HybridSearchService: Configured hybrid search service

        Raises:
            ValidationError: If weights are invalid
        """
        # Create keyword search if not provided
        if keyword_search is None:
            keyword_search = KeywordSearch()

        # Validate weights
        if not (0.0 <= vector_weight <= 1.0):
            raise ValidationError(
                f"vector_weight must be between 0.0 and 1.0, got {vector_weight}"
            )

        return HybridSearchService(
            vector_client=vector_client,
            keyword_search=keyword_search,
            vector_weight=vector_weight,
        )

    @staticmethod
    def create_reranker(
        reranker_type: str = "llm",
        llm_service: Optional[LLMServiceBase] = None,
        top_n: int = 5,
        model: Optional[str] = None,
        device: str = "cpu",
    ) -> object:
        """
        Create a reranking service.

        Args:
            reranker_type: Type of reranker ("cross_encoder", "llm", "noop")
            llm_service: LLM service (required for "llm" type)
            top_n: Number of top results to return
            model: Cross-encoder model name (for "cross_encoder" type)
            device: Device for cross-encoder model ("cpu" or "cuda")

        Returns:
            Reranking service instance

        Raises:
            ValidationError: If reranker_type is invalid or LLM not provided
        """
        if reranker_type == "noop":
            return NoOpReranker()

        elif reranker_type == "cross_encoder":
            return CrossEncoderReranker(
                model=model or "cross-encoder/ms-marco-MiniLM-L-6-v2",
                device=device,
                top_n=top_n,
            )

        elif reranker_type == "llm":
            if llm_service is None:
                raise ValidationError(
                    "llm_service is required for LLM-based reranking"
                )

            return RerankingService(
                llm_service=llm_service,
                top_n=top_n,
            )

        else:
            valid_types = ["cross_encoder", "llm", "noop"]
            raise ValidationError(
                f"Invalid reranker_type: {reranker_type}. "
                f"Must be one of {valid_types}"
            )

    @staticmethod
    def create_reranker_from_settings(
        llm_service: Optional[LLMServiceBase] = None,
    ) -> object:
        """
        Create a reranker from application settings.

        Supports single-stage (cross_encoder or llm) and two-stage (chained)
        reranking via RERANKER_TYPE and RERANKER_LLM_SECOND_STAGE settings.

        Args:
            llm_service: LLM service (required for LLM or chained with LLM second stage)

        Returns:
            Reranking service instance
        """
        from app.config.settings import get_settings

        settings = get_settings()

        if not settings.RERANKER_ENABLED:
            return NoOpReranker()

        reranker_type = settings.RERANKER_TYPE

        if reranker_type == "cross_encoder":
            reranker = CrossEncoderReranker(
                model=settings.RERANKER_MODEL,
                device=settings.RERANKER_DEVICE,
                top_n=settings.RERANKER_TOP_N,
            )
            if settings.RERANKER_LLM_SECOND_STAGE and llm_service:
                second_stage = RerankingService(
                    llm_service=llm_service,
                    top_n=settings.RERANKER_LLM_TOP_N,
                )
                return ChainedReranker(first_stage=reranker, second_stage=second_stage)
            return reranker

        elif reranker_type == "llm":
            if llm_service is None:
                raise ValidationError(
                    "llm_service is required for LLM-based reranking"
                )
            return RerankingService(
                llm_service=llm_service,
                top_n=settings.RERANKER_TOP_N,
            )

        elif reranker_type == "noop":
            return NoOpReranker()

        else:
            raise ValidationError(
                f"Invalid RERANKER_TYPE: {reranker_type}. "
                f"Must be one of: cross_encoder, llm, noop"
            )

    @staticmethod
    def create_pipeline(
        vector_client: VectorClient,
        llm_service: Optional[LLMServiceBase] = None,
        use_reranking: bool = True,
        vector_weight: float = 0.5,
        reranker_top_n: int = 5,
    ) -> dict:
        """
        Create a complete retrieval pipeline.

        Args:
            vector_client: Vector search client
            llm_service: Optional LLM service (for reranking)
            use_reranking: Whether to use LLM reranking
            vector_weight: Weight for vector search in hybrid (0.0-1.0)
            reranker_top_n: Number of results to return after reranking

        Returns:
            dict: Pipeline components
                - "vector_client": Vector client
                - "hybrid_search": Hybrid search service
                - "reranker": Reranking service
        """
        # Create hybrid search
        hybrid_search = RetrievalFactory.create_hybrid_search(
            vector_client=vector_client,
            vector_weight=vector_weight,
        )

        # Create reranker
        if use_reranking and llm_service:
            reranker = RetrievalFactory.create_reranker(
                reranker_type="llm",
                llm_service=llm_service,
                top_n=reranker_top_n,
            )
        else:
            reranker = RetrievalFactory.create_reranker(reranker_type="noop")

        return {
            "vector_client": vector_client,
            "hybrid_search": hybrid_search,
            "reranker": reranker,
        }
