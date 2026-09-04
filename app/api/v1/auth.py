"""
认证 API 端点。

提供用户注册、登录、令牌刷新与获取当前用户信息的端点。
"""
from typing import Annotated, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm

from app.models.schemas.user import (
    UserCreate,
    UserResponse,
    UserLogin,
    TokenResponse,
    RefreshTokenRequest,
)
from app.api.deps import (
    get_auth_service,
    get_current_user,
)
from app.services.auth_service import AuthenticationService
from app.models.database.user import User

router = APIRouter(prefix="/auth", tags=["认证"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="注册新用户",
)
async def register(
    user_data: UserCreate,
    auth_service: Annotated[AuthenticationService, Depends(get_auth_service)],
) -> User:
    """
    注册新用户账号。

    - **email**: 用户邮箱（必须唯一）
    - **password**: 密码（至少 8 位，须包含大写字母、小写字母与数字）
    - **full_name**: 用户全名（可选）

    返回创建的用户信息。
    """
    try:
        user = await auth_service.register_user(
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name,
        )
        return user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"注册失败: {str(e)}",
        ) from e


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="邮箱密码登录",
)
async def login(
    user_credentials: UserLogin,
    auth_service: Annotated[AuthenticationService, Depends(get_auth_service)],
) -> Dict[str, str | int]:
    """
    认证用户并返回访问令牌与刷新令牌。

    - **email**: 用户邮箱
    - **password**: 用户密码

    返回 JWT 访问令牌与刷新令牌。
    """
    try:
        # 认证用户
        user = await auth_service.authenticate_user(
            email=user_credentials.email,
            password=user_credentials.password,
        )

        # 创建令牌
        token_data = await auth_service.create_access_token(user)
        refresh_token = await auth_service.create_refresh_token(user)

        return {
            **token_data,
            "refresh_token": refresh_token,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="邮箱或密码错误",
        ) from e


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="刷新访问令牌",
)
async def refresh_token(
    token_data: RefreshTokenRequest,
    auth_service: Annotated[AuthenticationService, Depends(get_auth_service)],
) -> Dict[str, str | int]:
    """
    使用刷新令牌换取新的访问令牌。

    - **refresh_token**: 登录时获得的刷新令牌

    返回新的访问令牌。
    """
    try:
        new_token_data = await auth_service.refresh_access_token(
            refresh_token=token_data.refresh_token
        )
        return new_token_data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="刷新令牌无效或已过期",
        ) from e


@router.get(
    "/me",
    response_model=UserResponse,
    summary="获取当前用户信息",
)
async def get_current_user_info(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """
    获取当前登录用户的信息。

    需要在 Authorization 请求头携带有效的访问令牌。
    """
    return current_user
