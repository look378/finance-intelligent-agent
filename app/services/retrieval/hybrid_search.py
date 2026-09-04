"""
Hybrid search service combining vector and keyword search.

Implements Reciprocal Rank Fusion (RRF) to combine results from
vector similarity search and keyword-based search.
"""
from typing import Optional, List, Dict
import re
from collections import defaultdict

from app.services.retrieval.vector_base import (
    VectorClient,
    VectorSearchRequest,
    SearchResult,
    VectorClientError,
)
from app.core.exceptions import ValidationError


class KeywordSearch:
    """
    Simple keyword-based search implementation.

    Uses BM25-style ranking based on term frequency.
    """

    def __init__(self) -> None:
        """Initialize keyword search."""
        self.documents: Dict[str, dict] = {}
        self.document_terms: Dict[str, set[str]] = {}

    async def add_documents(
        self,
        documents: List[dict],
    ) -> None:
        """
        Add documents to keyword search index.

        Args:
            documents: List of dicts with 'id' and 'content' keys
        """
        for doc in documents:
            doc_id = doc["id"]
            content = doc["content"]
            self.documents[doc_id] = doc
            self.document_terms[doc_id] = self._extract_terms(content)

    async def search(
        self,
        request: VectorSearchRequest,
    ) -> List[SearchResult]:
        """
        Search documents by keyword matching.

        Args:
            request: Search request

        Returns:
            List[SearchResult]: Ranked search results
        """
        if not request.query.strip():
            # Return all documents with zero score for empty query
            return [
                SearchResult(
                    document_id=doc_id,
                    content=doc["content"],
                    score=0.0,
                )
                for doc_id, doc in self.documents.items()
            ]

        # Extract query terms
        query_terms = self._extract_terms(request.query)

        # Score each document
        scores = []
        for doc_id, terms in self.document_terms.items():
            score = self._compute_bm25_score(query_terms, terms)
            if score > 0:
                scores.append(
                    SearchResult(
                        document_id=doc_id,
                        content=self.documents[doc_id]["content"],
                        score=score,
                    )
                )

        # Sort by score (descending)
        scores.sort(key=lambda x: x.score, reverse=True)

        # Apply top_k limit
        return scores[: request.top_k]

    def _extract_terms(self, text: str) -> set[str]:
        """
        Extract search terms from text.

        Args:
            text: Input text

        Returns:
            set[str]: Set of lowercase terms
        """
        # Convert to lowercase and extract alphanumeric terms
        terms = re.findall(r"\b\w+\b", text.lower())
        return set(terms)

    def _compute_bm25_score(
        self,
        query_terms: set[str],
        doc_terms: set[str],
    ) -> float:
        """
        Compute BM25-style score.

        Args:
            query_terms: Query term set
            doc_terms: Document term set

        Returns:
            float: BM25 score
        """
        if not query_terms or not doc_terms:
            return 0.0

        # Count matching terms
        matches = query_terms.intersection(doc_terms)

        if not matches:
            return 0.0

        # Simple score: number of matching terms / query length
        # This is a simplified BM25 (without IDF and document length normalization)
        return len(matches) / len(query_terms)


class HybridSearchService:
    """
    Hybrid search service combining vector and keyword search.

    Uses Reciprocal Rank Fusion (RRF) to combine rankings from
    vector similarity search and keyword-based search.
    """

    def __init__(
        self,
        vector_client: VectorClient,
        keyword_search: KeywordSearch,
        vector_weight: float = 0.5,
    ) -> None:
        """
        Initialize hybrid search service.

        Args:
            vector_client: Vector search client
            keyword_search: Keyword search instance
            vector_weight: Weight for vector search (0.0-1.0)

        Raises:
            ValidationError: If weights don't sum to 1.0
        """
        keyword_weight = 1.0 - vector_weight

        if not (0.0 <= vector_weight <= 1.0):
            raise ValidationError(
                f"vector_weight must be between 0.0 and 1.0, got {vector_weight}"
            )

        self.vector_client = vector_client
        self.keyword_search = keyword_search
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight

    async def search(
        self,
        request: VectorSearchRequest,
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and keyword results.

        Args:
            request: Search request

        Returns:
            List[SearchResult]: Combined and reranked results

        Raises:
            VectorClientError: If both search methods fail
        """
        vector_results = []
        keyword_results = []

        # Try vector search
        try:
            vector_results = await self.vector_client.search(request)
        except Exception as e:
            # Log error but continue with keyword-only
            pass

        # Try keyword search
        try:
            keyword_results = await self.keyword_search.search(request)
        except Exception as e:
            # Log error but continue with vector-only
            pass

        # If both failed, raise error
        if not vector_results and not keyword_results:
            raise VectorClientError(
                "Both vector and keyword search failed"
            )

        # Combine results using RRF
        combined = self._reciprocal_rank_fusion(
            vector_results,
            keyword_results,
        )

        # Apply top_k limit
        return combined[: request.top_k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
        k: int = 60,
    ) -> List[SearchResult]:
        """
        Combine rankings using Reciprocal Rank Fusion (RRF).

        RRF formula: score = sum(1 / (k + rank))

        Args:
            vector_results: Vector search results
            keyword_results: Keyword search results
            k: RRF constant (default: 60)

        Returns:
            List[SearchResult]: Fused and reranked results
        """
        # Accumulate RRF scores
        scores: Dict[str, float] = defaultdict(float)
        doc_data: Dict[str, dict] = {}

        # Process vector results
        for rank, result in enumerate(vector_results, start=1):
            doc_id = result.document_id
            rrf_score = 1.0 / (k + rank)
            scores[doc_id] += self.vector_weight * rrf_score

            # Store document data
            if doc_id not in doc_data:
                doc_data[doc_id] = {
                    "content": result.content,
                    "metadata": result.metadata or {},
                    "vector_score": result.score,
                    "keyword_score": 0.0,
                }

        # Process keyword results
        for rank, result in enumerate(keyword_results, start=1):
            doc_id = result.document_id
            rrf_score = 1.0 / (k + rank)
            scores[doc_id] += self.keyword_weight * rrf_score

            # Store or update document data
            if doc_id not in doc_data:
                doc_data[doc_id] = {
                    "content": result.content,
                    "metadata": result.metadata or {},
                    "vector_score": 0.0,
                    "keyword_score": result.score,
                }
            else:
                doc_data[doc_id]["keyword_score"] = result.score

        # Build final results
        fused_results = []
        for doc_id, score in scores.items():
            data = doc_data[doc_id]

            # Store individual scores in metadata
            metadata = data["metadata"].copy() if data["metadata"] else {}
            metadata["vector_score"] = data["vector_score"]
            metadata["keyword_score"] = data["keyword_score"]
            metadata["rrf_score"] = score

            result = SearchResult(
                document_id=doc_id,
                content=data["content"],
                score=score,
                metadata=metadata,
            )
            fused_results.append(result)

        # Sort by RRF score (descending)
        fused_results.sort(key=lambda x: x.score, reverse=True)

        return fused_results
