"""反馈 API 的 Pydantic 模型。"""
from pydantic import BaseModel, Field


class FeedbackCreate(BaseModel):
    """提交反馈的请求体。"""
    message_id: int = Field(..., description="要评分的助手消息 ID")
    rating: int = Field(..., ge=-1, le=1, description="评分：1=点赞，-1=点踩")
    text: str | None = Field(None, max_length=1000, description="可选反馈文本")


class FeedbackResponse(BaseModel):
    """提交反馈后的响应。"""
    success: bool
    message_id: int
    rating: int
