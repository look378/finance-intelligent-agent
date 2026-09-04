"""Feedback repository for message rating operations."""
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database.message import Message


class FeedbackRepository:
    """Async repository for feedback operations on messages."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def submit_feedback(
        self,
        message_id: int,
        rating: int,
        text: Optional[str] = None,
    ) -> Optional[Message]:
        """Update a message with user feedback. Returns the message or None if not found."""
        message = await self._db.get(Message, message_id)
        if message is None:
            return None

        message.user_rating = rating
        message.feedback_text = text
        await self._db.commit()
        await self._db.refresh(message)
        return message

    async def get_feedback_stats(self) -> dict:
        """Get aggregate feedback statistics."""
        result = await self._db.execute(
            select(Message.user_rating).where(Message.user_rating.isnot(None))
        )
        ratings = [r for (r,) in result.fetchall()]
        if not ratings:
            return {"total": 0, "positive": 0, "negative": 0, "average": 0.0}
        return {
            "total": len(ratings),
            "positive": sum(1 for r in ratings if r > 0),
            "negative": sum(1 for r in ratings if r < 0),
            "average": sum(ratings) / len(ratings),
        }
