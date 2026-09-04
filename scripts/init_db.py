#!/usr/bin/env python3
"""
Database initialization script for production deployment.

Creates all database tables and default data automatically.
This script is called by the container entrypoint on first startup.
"""
import asyncio
import sys
from pathlib import Path

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.models.database.base import Base
from app.models.database.user import User
from app.models.database.session import ChatSession
from app.models.database.message import Message
from app.models.database.document import Document
from app.config.logging import logger
from app.config.settings import settings

# Create engine for initialization
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
)

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def create_tables():
    """Create all database tables."""
    async with engine.begin() as conn:
        # Import all models to ensure they are registered with Base
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")


async def create_demo_session():
    """Create a default demo session for anonymous users."""
    # Use the async_session_maker defined above

    async with async_session_maker() as session:
        # Check if demo session exists
        result = await session.execute(
            text("SELECT id FROM chat_sessions WHERE id = 1")
        )
        if result.first() is None:
            # Create demo user first
            from app.core.security import hash_password

            # Check if demo user exists
            user_result = await session.execute(
                text("SELECT id FROM users WHERE email = 'demo@finance_agent.com'")
            )

            demo_user_id = None
            if user_result.first() is None:
                # Create demo user
                demo_user = User(
                    email="demo@finance_agent.com",
                    full_name="Demo User",
                    hashed_password=hash_password("demo123"),
                    is_active=True,
                    is_admin=False,
                )
                session.add(demo_user)
                await session.flush()
                demo_user_id = demo_user.id
                logger.info("Demo user created")
            else:
                demo_user_id = user_result.first()[0]
                logger.info("Demo user already exists")

            # Create demo session
            demo_session = ChatSession(
                id=1,
                user_id=demo_user_id,
                title="Demo Chat Session",
                memory_type="sliding_window",
                context_window=10,
            )
            session.add(demo_session)
            await session.commit()
            logger.info("Demo session created with ID=1")
        else:
            logger.info("Demo session already exists")


async def verify_database():
    """Verify database is ready."""
    try:
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Database connection verified")
        return True
    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        return False


async def main():
    """Main initialization function."""
    logger.info("Starting database initialization...")

    # Wait for database to be ready
    max_retries = 30
    for i in range(max_retries):
        if await verify_database():
            break
        logger.info(f"Database not ready, retrying... ({i+1}/{max_retries})")
        await asyncio.sleep(2)
    else:
        logger.error("Database connection failed after max retries")
        sys.exit(1)

    # Create tables
    await create_tables()

    # Create demo data
    await create_demo_session()

    logger.info("Database initialization complete")


if __name__ == "__main__":
    asyncio.run(main())
