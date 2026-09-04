"""会话相关 Pydantic 模型"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List
from datetime import datetime


class SessionBase(BaseModel):
    """会话基础模型"""
    title: Optional[str] = Field(None, max_length=255, description="会话标题")
    memory_type: str = Field("sliding_window", pattern="^(sliding_window|summarization|hybrid)$", description="记忆策略")
    context_window: int = Field(10, ge=1, le=100, description="上下文消息数")


class SessionCreate(SessionBase):
    """创建新会话模型"""
    pass


class SessionUpdate(BaseModel):
    """更新会话模型"""
    title: Optional[str] = Field(None, max_length=255, description="会话标题")
    memory_type: Optional[str] = Field(None, pattern="^(sliding_window|summarization|hybrid)$", description="记忆策略")
    context_window: Optional[int] = Field(None, ge=1, le=100, description="上下文消息数")


class SessionResponse(SessionBase):
    """会话响应模型"""
    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    model_config = ConfigDict(from_attributes=True)


class SessionListResponse(BaseModel):
    """分页会话列表模型"""
    items: List[SessionResponse]
    total: int
    page: int
    page_size: int
