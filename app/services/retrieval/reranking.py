"""
Result reranking service for improving search results.

Provides LLM-based and no-op reranking strategies to optimize
search result ordering based on query relevance and diversity.
"""
from typing import Optional, List
import json

from app.services.retrieval.vector_base import (
    VectorSearchRequest,
    SearchResult,
)
from app.services.llm.base import LLMServiceBase, LLMMessage


class NoOpReranker:
    """
    No-op reranker that passes results through unchanged.

    Useful for disabling reranking while maintaining the same interface.
    """

    async def rerank(
        self,
        results: List[SearchResult],
        request: VectorSearchRequest,
    ) -> List[SearchResult]:
        """
        Return results unchanged (no reranking).

        Args:
            results: Search results to rerank
            request: Original search request

        Returns:
            List[SearchResult]: Same results (unchanged)
        """
        return results


class RerankingService:
    """
    LLM-based reranking service.

    Uses LLM to rerank search results based on query relevance
    and result diversity.
    """

    def __init__(
        self,
        llm_service: LLMServiceBase,
        top_n: int = 5,
    ) -> None:
        """
        Initialize reranking service.

        Args:
            llm_service: LLM service for reranking
            top_n: Number of top results to return
        """
        self.llm_service = llm_service
        self.top_n = top_n

    async def rerank(
        self,
        results: List[SearchResult],
        request: VectorSearchRequest,
    ) -> List[SearchResult]:
        """
        Rerank search results using LLM.

        Args:
            results: Search results to rerank
            request: Original search request

        Returns:
            List[SearchResult]: Reranked results (limited to top_n)
        """
        if not results:
            return []

        # Build reranking prompt
        prompt = self._build_reranking_prompt(
            query=request.query,
            results=results,
        )

        # Call LLM
        response = await self.llm_service.generate(
            messages=[
                LLMMessage(
                    role="system",
                    content="You are a search result reranking system. "
                    "Return JSON with reranked document IDs and optionally scores.",
                ),
                LLMMessage(role="user", content=prompt),
            ],
            max_tokens=500,
            temperature=0.0,  # Deterministic
        )

        # Parse LLM response
        return self._parse_reranking_response(
            response.content,
            original_results=results,
        )

    def _build_reranking_prompt(
        self,
        query: str,
        results: List[SearchResult],
    ) -> str:
        """
        Build prompt for LLM reranking.

        Args:
            query: Search query
            results: Search results

        Returns:
            str: Reranking prompt
        """
        # Format results for prompt
        result_docs = []
        for i, result in enumerate(results, start=1):
            result_docs.append(
                f"{i}. Document ID: {result.document_id}\n"
                f"   Content: {result.content}\n"
                f"   Score: {result.score:.3f}\n"
            )

        results_text = "\n".join(result_docs)

        prompt = f"""Given this search query:

"{query}"

And these search results (ranked by initial similarity score):

{results_text}

Please rerank the results based on:
1. Relevance to the query
2. Information quality
3. Diversity of content

Return a JSON object with this format:
{{"reranked": ["doc_id1", "doc_id2", "doc_id3", ...]}}

Or if you want to provide new relevance scores:
{{"results": [{{"id": "doc_id1", "score": 0.95}}, {{"id": "doc_id2", "score": 0.85}}, ...]}}

Only rerank the top {self.top_n} results."""

        return prompt

    def _parse_reranking_response(
        self,
        response: str,
        original_results: List[SearchResult],
    ) -> List[SearchResult]:
        """
        Parse LLM reranking response.

        Args:
            response: LLM response text
            original_results: Original search results

        Returns:
            List[SearchResult]: Reranked results
        """
        try:
            # Parse JSON response
            data = json.loads(response.strip())

            # Build lookup map
            result_map = {r.document_id: r for r in original_results}

            reranked = []

            # Case 1: Simple reranked list
            if "reranked" in data:
                doc_ids = data["reranked"]

                for doc_id in doc_ids:
                    if doc_id in result_map:
                        result = result_map[doc_id]

                        # Add metadata
                        metadata = result.metadata or {}
                        metadata["reranked"] = True

                        reranked.append(
                            SearchResult(
                                document_id=result.document_id,
                                content=result.content,
                                score=result.score,
                                metadata=metadata,
                            )
                        )

                # Add remaining results that weren't in reranked list
                for result in original_results:
                    if result.document_id not in doc_ids:
                        reranked.append(result)

            # Case 2: Results with new scores
            elif "results" in data:
                scored_results = data["results"]

                for item in scored_results:
                    doc_id = item["id"]
                    new_score = item.get("score", 0.0)

                    if doc_id in result_map:
                        result = result_map[doc_id]

                        # Add metadata
                        metadata = result.metadata or {}
                        metadata["reranked"] = True
                        metadata["reranking_score"] = new_score

                        reranked.append(
                            SearchResult(
                                document_id=result.document_id,
                                content=result.content,
                                score=new_score,
                                metadata=metadata,
                            )
                        )

            else:
                # Invalid response format, return original
                return original_results

            # Limit to top_n
            return reranked[: self.top_n]

        except (json.JSONDecodeError, KeyError, ValueError):
            # Parse error, return original results
            return original_results
