"""用户相关 Pydantic 模型"""
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from datetime import datetime
from typing import Optional


class UserBase(BaseModel):
    """用户基础模型（公共字段）"""
    email: EmailStr
    full_name: Optional[str] = None


class UserCreate(UserBase):
    """用户注册模型"""
    password: str = Field(..., min_length=8, max_length=100, description="密码（至少8位）")


class UserLogin(BaseModel):
    """用户登录模型"""
    email: EmailStr
    password: str = Field(..., min_length=1)


class UserResponse(UserBase):
    """用户响应模型"""
    id: int
    is_active: bool
    is_admin: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TokenResponse(BaseModel):
    """认证令牌响应模型"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None


class RefreshTokenRequest(BaseModel):
    """令牌刷新请求模型"""
    refresh_token: str


class UserUpdate(BaseModel):
    """用户信息更新模型"""
    full_name: Optional[str] = None
    email: Optional[EmailStr] = None


class ChangePasswordRequest(BaseModel):
    """修改密码请求模型"""
    old_password: str
    new_password: str = Field(..., min_length=8, max_length=100)
