"""消息评分反馈 API 端点。"""
import logging

from fastapi import APIRouter, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schemas.feedback import FeedbackCreate, FeedbackResponse
from app.repositories.feedback_repository import FeedbackRepository
from app.api.deps import get_current_user
from app.models.database.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/feedback", tags=["反馈"])


async def _get_db():
    from app.api.database import async_session_maker
    async with async_session_maker() as session:
        yield session


@router.post("", response_model=FeedbackResponse, status_code=status.HTTP_200_OK, summary="提交消息反馈")
async def submit_feedback(
    feedback: FeedbackCreate,
    db: AsyncSession = Depends(_get_db),
    current_user: User = Depends(get_current_user),
):
    """为助手消息提交反馈（点赞/点踩）。"""
    repo = FeedbackRepository(db)
    message = await repo.submit_feedback(
        message_id=feedback.message_id,
        rating=feedback.rating,
        text=feedback.text,
    )
    if message is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"消息 {feedback.message_id} 不存在",
        )
    return FeedbackResponse(
        success=True,
        message_id=message.id,
        rating=message.user_rating or 0,
    )


@router.get("/stats", summary="获取反馈统计")
async def get_feedback_stats(
    db: AsyncSession = Depends(_get_db),
    current_user: User = Depends(get_current_user),
):
    """获取聚合后的反馈统计数据。"""
    repo = FeedbackRepository(db)
    return await repo.get_feedback_stats()
