"""
FastAPI dependencies for dependency injection.

Provides common dependencies for endpoints.
"""
from typing import AsyncGenerator, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_access_token
from app.core.exceptions import AuthenticationError, NotFoundError
from app.models.database.user import User
from app.repositories.user_repository import UserRepository
from app.repositories.session_repository import SessionRepository
from app.services.auth_service import AuthenticationService

# HTTP Bearer token scheme - auto_error=False for demo mode (allows anonymous access)
security = HTTPBearer(auto_error=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session.

    Yields:
        AsyncSession: Database session
    """
    from app.api.database import get_db as _get_db
    async for session in _get_db():
        yield session


async def get_user_repository(
    db: AsyncSession = Depends(get_db),
) -> UserRepository:
    """
    Dependency to get user repository.

    Args:
        db: Database session

    Returns:
        UserRepository: User repository instance
    """
    return UserRepository(db)


async def get_auth_service(
    user_repo: UserRepository = Depends(get_user_repository),
) -> AuthenticationService:
    """
    Dependency to get authentication service.

    Args:
        user_repo: User repository

    Returns:
        AuthenticationService: Authentication service instance
    """
    return AuthenticationService(user_repo)


async def get_session_repository(
    db: AsyncSession = Depends(get_db),
) -> SessionRepository:
    """
    Dependency to get session repository.

    Args:
        db: Database session

    Returns:
        SessionRepository: Session repository instance
    """
    return SessionRepository(db)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth_service: AuthenticationService = Depends(get_auth_service),
) -> Optional[User]:
    """
    Dependency to get the current authenticated user from JWT token.

    DEMO MODE: Returns None if no credentials provided (allows anonymous access).

    Args:
        credentials: Optional HTTP Bearer credentials
        auth_service: Authentication service

    Returns:
        Optional[User]: Current authenticated user, or None if not authenticated

    Raises:
        HTTPException: If authentication token is invalid (but not if missing)
    """
    # Demo mode: allow anonymous access
    if credentials is None:
        return None

    try:
        token = credentials.credentials
        payload = await auth_service.verify_token(token)

        user_id = payload.get("user_id")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )

        user_repo = auth_service.user_repository
        user = await user_repo.get_by_id(user_id, User)

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is inactive",
            )

        return user

    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """
    Dependency to get the current active user.

    Args:
        current_user: Current authenticated user

    Returns:
        User: Current active user

    Raises:
        HTTPException: If user is not active
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )
    return current_user


async def require_admin(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """
    Dependency to require admin role.

    Args:
        current_user: Current authenticated user

    Returns:
        User: Current admin user

    Raises:
        HTTPException: If user is not an admin
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return current_user
