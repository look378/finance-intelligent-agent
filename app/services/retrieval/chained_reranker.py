"""
Chained reranker for two-stage reranking.

Orchestrates a first-stage reranker (e.g., CrossEncoder for coarse ranking)
and an optional second-stage reranker (e.g., LLM for fine ranking).
"""
from typing import List, Optional

from app.services.retrieval.vector_base import SearchResult, VectorSearchRequest


class ChainedReranker:
    """
    Two-stage reranker that chains a first-stage and optional second-stage reranker.

    Both stages share the same duck-typed interface:
        async def rerank(results, request) -> List[SearchResult]
    """

    def __init__(self, first_stage: object, second_stage: Optional[object] = None) -> None:
        self.first_stage = first_stage
        self.second_stage = second_stage

    async def rerank(
        self,
        results: List[SearchResult],
        request: VectorSearchRequest,
    ) -> List[SearchResult]:
        """
        Run first-stage reranking, then optionally second-stage.

        Args:
            results: Search results to rerank
            request: Original search request

        Returns:
            Reranked results after all stages.
        """
        reranked = await self.first_stage.rerank(results, request)

        if self.second_stage and reranked:
            reranked = await self.second_stage.rerank(reranked, request)

        return reranked
