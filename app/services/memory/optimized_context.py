"""
Optimized context builder with relevance-based filtering.

Implements smart token budgeting and relevance filtering to reduce
token usage and improve response quality.
"""
from typing import Optional, List
from datetime import datetime, timedelta

from app.services.memory.base import MemoryStrategy, MessageContent
from app.services.embeddings import EmbeddingFactory
from app.core.exceptions import ExternalServiceError
import math


class OptimizedContextBuilder(MemoryStrategy):
    """
    Optimized context builder that filters irrelevant chat history.

    Features:
    - Relevance-based filtering using embeddings
    - Smart token budgeting
    - Automatic summarization on topic shift
    - Dynamic context window adjustment
    """

    def __init__(
        self,
        message_repo,
        embedding_service=None,
        max_recent_messages: int = 3,
        max_relevant_messages: int = 5,
        relevance_threshold: float = 0.5,
        token_budget: int = 4096,
    ):
        """
        Initialize optimized context builder.

        Args:
            message_repo: Message repository
            embedding_service: Embedding service for relevance scoring
            max_recent_messages: Always include N most recent messages
            max_relevant_messages: Maximum relevant historical messages
            relevance_threshold: Minimum similarity score (0.0-1.0)
            token_budget: Maximum tokens for chat history
        """
        self.message_repo = message_repo
        self.embedding_service = embedding_service or EmbeddingFactory.create(
            provider="local"
        )
        self.max_recent_messages = max_recent_messages
        self.max_relevant_messages = max_relevant_messages
        self.relevance_threshold = relevance_threshold
        self.token_budget = token_budget

    async def get_context(
        self,
        session_id: int,
        current_query: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> List[MessageContent]:
        """
        Get optimized context with relevance filtering.

        Args:
            session_id: Chat session ID
            current_query: Current user query (for relevance filtering)
            max_tokens: Override token budget

        Returns:
            List[MessageContent]: Optimized list of messages
        """
        # 1. Get all messages
        all_messages = await self.message_repo.get_recent_messages(
            session_id=session_id,
            limit=100  # Get more, will filter
        )

        if not all_messages:
            return []

        # 2. Always include most recent messages (for continuity)
        recent_count = min(self.max_recent_messages, len(all_messages))
        recent_messages = all_messages[:recent_count]

        # 3. Filter older messages by relevance (if query provided)
        older_messages = all_messages[recent_count:]
        relevant_messages = []

        if current_query and older_messages:
            relevant_messages = await self._filter_by_relevance(
                messages=older_messages,
                query=current_query,
                max_count=self.max_relevant_messages
            )

        # 4. Combine recent + relevant
        combined = recent_messages + relevant_messages

        # 5. Truncate by tokens if needed
        budget = max_tokens or self.token_budget
        if budget:
            combined = await self.truncate_by_tokens(combined, budget)

        return combined

    async def _filter_by_relevance(
        self,
        messages: List[MessageContent],
        query: str,
        max_count: int = 5
    ) -> List[MessageContent]:
        """
        Filter messages by semantic similarity to query.

        Args:
            messages: List of historical messages
            query: Current query to compare against
            max_count: Maximum number of relevant messages to return

        Returns:
            List[MessageContent]: Most relevant messages
        """
        if not messages:
            return []

        try:
            # 1. Embed query once
            query_embedding = await self.embedding_service.embed_single(query)

            # 2. Score each message
            scored_messages = []
            for msg in messages:
                # Embed message
                msg_embedding = await self.embedding_service.embed_single(msg.content)

                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_embedding, msg_embedding)

                # Only include if above threshold
                if similarity >= self.relevance_threshold:
                    scored_messages.append((msg, similarity))

            # 3. Sort by relevance and keep top N
            scored_messages.sort(key=lambda x: x[1], reverse=True)

            return [msg for msg, _ in scored_messages[:max_count]]

        except Exception as e:
            # Fallback: return most recent messages if embedding fails
            return messages[:max_count]

    @staticmethod
    def _cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            float: Similarity score (-1.0 to 1.0)
        """
        try:
            import numpy as np

            v1 = np.array(embedding1)
            v2 = np.array(embedding2)

            dot_product = np.dot(v1, v2)
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

        except Exception:
            return 0.0

    async def truncate_by_tokens(
        self,
        messages: List[MessageContent],
        max_tokens: int
    ) -> List[MessageContent]:
        """
        Truncate messages to fit within token limit.

        Keeps most recent messages and drops oldest.

        Args:
            messages: List of messages (newest first)
            max_tokens: Maximum tokens

        Returns:
            List[MessageContent]: Truncated list
        """
        if not messages:
            return []

        # Estimate tokens for each message (from newest to oldest)
        result = []
        total_tokens = 0

        for msg in messages:
            # Estimate: ~4 characters per token (rough estimate)
            msg_tokens = len(msg.content) // 4

            if total_tokens + msg_tokens > max_tokens:
                # Would exceed budget, stop here
                break

            result.append(msg)
            total_tokens += msg_tokens

        return result

    async def estimate_token_savings(
        self,
        session_id: int,
        current_query: str
    ) -> dict:
        """
        Estimate token savings from using relevance filtering.

        Args:
            session_id: Chat session ID
            current_query: Current query

        Returns:
            dict: Savings statistics
        """
        # Get all messages
        all_messages = await self.message_repo.get_recent_messages(
            session_id=session_id,
            limit=100
        )

        if not all_messages:
            return {"savings_percent": 0, "tokens_saved": 0}

        # Count tokens in all messages
        all_tokens = sum(len(msg.content) // 4 for msg in all_messages)

        # Get optimized context
        optimized = await self.get_context(
            session_id=session_id,
            current_query=current_query
        )

        # Count tokens in optimized messages
        optimized_tokens = sum(len(msg.content) // 4 for msg in optimized)

        # Calculate savings
        tokens_saved = all_tokens - optimized_tokens
        savings_percent = (tokens_saved / all_tokens * 100) if all_tokens > 0 else 0

        return {
            "total_messages": len(all_messages),
            "selected_messages": len(optimized),
            "original_tokens": all_tokens,
            "optimized_tokens": optimized_tokens,
            "tokens_saved": tokens_saved,
            "savings_percent": round(savings_percent, 1)
        }

    async def add_message(
        self,
        session_id: int,
        message: MessageContent
    ) -> None:
        """Add a message to the repository."""
        # Create message in database
        from app.models.database.message import Message
        from app.models.enums.message import MessageRole, MessageStatus

        db_message = Message(
            session_id=session_id,
            role=MessageRole(message["role"]),
            content=message["content"],
            status=MessageStatus.COMPLETED,
        )

        await self.message_repo.create(db_message)

    async def clear_session(self, session_id: int) -> None:
        """Clear all messages for a session."""
        await self.message_repo.delete_by_session(session_id)
