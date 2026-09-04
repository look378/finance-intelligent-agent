"""
Cross-encoder reranking service using sentence-transformers.

Uses a lightweight cross-encoder model (e.g., ms-marco-MiniLM-L-6-v2)
for fast, cost-effective reranking on CPU.
"""
import asyncio
from typing import List

from app.services.retrieval.vector_base import SearchResult, VectorSearchRequest
from app.core.exceptions import ExternalServiceError


class CrossEncoderReranker:
    """
    Cross-encoder reranker using sentence-transformers.

    Lazy-loads the model on first use. Runs inference in a thread pool
    to avoid blocking the async event loop.
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        top_n: int = 5,
    ) -> None:
        self.model_name = model
        self.device = device
        self.top_n = top_n
        self._model = None
        self._load_lock = asyncio.Lock()

    async def _load_model(self) -> None:
        """Load cross-encoder model (lazy, thread-safe)."""
        if self._model is not None:
            return

        async with self._load_lock:
            if self._model is not None:
                return

            try:
                from sentence_transformers import CrossEncoder

                loop = asyncio.get_event_loop()
                self._model = await loop.run_in_executor(
                    None,
                    lambda: CrossEncoder(self.model_name, device=self.device),
                )
            except ImportError as e:
                raise ExternalServiceError(
                    service="CrossEncoderReranker",
                    message="sentence-transformers not installed. Run: pip install sentence-transformers",
                ) from e
            except Exception as e:
                raise ExternalServiceError(
                    service="CrossEncoderReranker",
                    message=f"Failed to load model {self.model_name}: {str(e)}",
                ) from e

    async def rerank(
        self,
        results: List[SearchResult],
        request: VectorSearchRequest,
    ) -> List[SearchResult]:
        """
        Rerank search results using cross-encoder scoring.

        Args:
            results: Search results to rerank
            request: Original search request (provides query)

        Returns:
            Top-N results sorted by cross-encoder relevance score.
        """
        if not results:
            return []

        await self._load_model()

        try:
            pairs = [(request.query, r.content) for r in results]

            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None,
                lambda: self._model.predict(pairs),
            )

            scored = list(zip(results, scores.tolist()))
            scored.sort(key=lambda x: x[1], reverse=True)

            reranked = []
            for result, score in scored[: self.top_n]:
                metadata = result.metadata or {}
                metadata["original_score"] = result.score
                metadata["reranker"] = "cross_encoder"
                reranked.append(
                    SearchResult(
                        document_id=result.document_id,
                        content=result.content,
                        score=score,
                        metadata=metadata,
                    )
                )

            return reranked

        except Exception as e:
            raise ExternalServiceError(
                service="CrossEncoderReranker",
                message=f"Reranking failed: {str(e)}",
            ) from e
