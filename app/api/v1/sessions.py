"""
会话 API 端点。

提供聊天会话的增删改查端点。
"""
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status, Query

from app.models.schemas.session import (
    SessionCreate,
    SessionUpdate,
    SessionResponse,
    SessionListResponse,
)
from app.api.deps import (
    get_session_repository,
    get_current_user,
)
from app.services.session_service import SessionService
from app.repositories.session_repository import SessionRepository
from app.models.database.user import User

router = APIRouter(prefix="/sessions", tags=["会话"])


async def get_session_service(
    session_repo: SessionRepository = Depends(get_session_repository),
) -> SessionService:
    """
    获取会话服务的依赖项。

    Args:
        session_repo: 会话仓库

    Returns:
        SessionService: 会话服务实例
    """
    return SessionService(session_repo)


@router.post(
    "",
    response_model=SessionResponse,
    status_code=status.HTTP_200_OK,
    summary="创建新的聊天会话",
)
async def create_session(
    session_data: SessionCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
) -> dict:
    """
    为当前登录用户创建新的聊天会话。

    - **title**: 会话标题（可选）
    - **memory_type**: 记忆策略（sliding_window、summarization 或 hybrid）
    - **context_window**: 上下文中保留的消息数（1-100）

    返回创建的会话。
    """
    try:
        session = await session_service.create_session(
            user_id=current_user.id,
            title=session_data.title,
            memory_type=session_data.memory_type,
            context_window=session_data.context_window,
        )
        return session
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建会话失败: {str(e)}",
        ) from e


@router.get(
    "",
    response_model=SessionListResponse,
    summary="查询用户的聊天会话列表",
)
async def list_sessions(
    current_user: Annotated[User, Depends(get_current_user)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
    page: int = Query(1, ge=1, description="页码"),
    page_size: int = Query(20, ge=1, le=100, description="每页条数"),
) -> dict:
    """
    分页查询当前登录用户的所有聊天会话。

    按最近更新时间倒序返回分页会话列表。
    """
    try:
        skip = (page - 1) * page_size
        sessions = await session_service.get_user_sessions(
            user_id=current_user.id,
            skip=skip,
            limit=page_size,
        )

        # 获取总数
        total = await session_service.count_user_sessions(current_user.id)

        return {
            "items": sessions,
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"查询会话列表失败: {str(e)}",
        ) from e


@router.get(
    "/{session_id}",
    response_model=SessionResponse,
    summary="获取指定聊天会话",
)
async def get_session(
    session_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
) -> dict:
    """
    按 ID 获取指定的聊天会话。

    仅返回属于当前登录用户的会话。
    """
    try:
        session = await session_service.get_session_by_id(session_id, current_user.id)
        return session
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"会话不存在: {str(e)}",
        ) from e


@router.put(
    "/{session_id}",
    response_model=SessionResponse,
    summary="更新聊天会话",
)
async def update_session(
    session_id: int,
    session_data: SessionUpdate,
    current_user: Annotated[User, Depends(get_current_user)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
) -> dict:
    """
    更新聊天会话信息。

    仅更新属于当前登录用户的会话。
    所有字段均为可选，仅更新传入的字段。
    """
    try:
        session = await session_service.update_session(
            session_id=session_id,
            user_id=current_user.id,
            title=session_data.title,
            memory_type=session_data.memory_type,
            context_window=session_data.context_window,
        )
        return session
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新会话失败: {str(e)}",
        ) from e


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除聊天会话",
)
async def delete_session(
    session_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    session_service: Annotated[SessionService, Depends(get_session_service)],
) -> None:
    """
    删除聊天会话。

    仅删除属于当前登录用户的会话。
    与该会话关联的所有消息也会一并删除。
    """
    try:
        await session_service.delete_session(session_id, current_user.id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除会话失败: {str(e)}",
        ) from e
