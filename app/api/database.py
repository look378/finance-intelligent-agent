"""
Database session management for FastAPI.

Provides async database session management with proper lifecycle handling.
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from app.config.settings import settings

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DATABASE_POOL_SIZE,
    max_overflow=settings.DATABASE_MAX_OVERFLOW,
    echo=settings.DEBUG,
)

# Create session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Get a database session with proper error handling.

    Yields:
        AsyncSession: Database session

    Example:
        async with get_db_session() as session:
            # Use session
            pass
    """
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting database session in FastAPI endpoints.

    This is the dependency that should be used with FastAPI's Depends.

    Yields:
        AsyncSession: Database session
    """
    async with async_session_maker() as session:
        yield session


async def init_db() -> None:
    """
    Initialize database connection.

    This can be called during application startup to verify
    database connectivity.
    """
    async with engine.begin() as conn:
        # Test connection
        await conn.execute("SELECT 1")


async def close_db() -> None:
    """
    Close database connections.

    This should be called during application shutdown.
    """
    await engine.dispose()
