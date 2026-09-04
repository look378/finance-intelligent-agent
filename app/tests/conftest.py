"""
Pytest configuration and fixtures.

This file contains shared fixtures and configuration for all tests.
"""
import asyncio
import pytest
from collections.abc import AsyncGenerator, Generator
from typing import AsyncGenerator as AsyncGeneratorType
from unittest.mock import AsyncMock, Mock

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import StaticPool
from httpx import AsyncClient, ASGITransport

from app.models.database.base import Base
from app.config.settings import settings
from app.main import create_app


# Test database URL (in-memory SQLite for speed)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop() -> asyncio.AbstractEventLoop:
    """
    Create an instance of the event loop for the test session.

    This fixture is necessary for async tests to work properly.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def test_engine():
    """
    Create a test database engine.

    Uses in-memory SQLite for fast test execution.
    """
    engine = create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    yield engine
    await engine.dispose()


@pytest.fixture(scope="function")
async def db_session(test_engine) -> AsyncGeneratorType[AsyncSession, None]:
    """
    Create a test database session.

    This fixture creates all tables before each test and drops them after.
    """
    # Create all tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Create session
    async_session_maker = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with async_session_maker() as session:
        yield session

    # Drop all tables after test
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
def mock_settings(monkeypatch):
    """
    Mock application settings for testing.

    Ensures tests don't depend on environment variables.
    """
    monkeypatch.setenv("ENVIRONMENT", "testing")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-for-testing-only")


@pytest.fixture
def mock_llm_service():
    """
    Mock LLM service for testing.

    Prevents actual API calls to LLM providers.
    """
    mock = AsyncMock()
    mock.generate.return_value = "This is a mock LLM response"
    return mock


@pytest.fixture
def mock_vector_service():
    """
    Mock vector database service for testing.

    Prevents actual calls to external vector service.
    """
    mock = AsyncMock()
    mock.search.return_value = [
        {
            "content": "Sample document chunk",
            "score": 0.95,
            "metadata": {"source": "test.pdf", "page": 1},
        },
    ]
    return mock


@pytest.fixture
def sample_user():
    """
    Create a sample user dictionary for testing.
    """
    return {
        "email": "test@example.com",
        "password": "hashed_password",
        "full_name": "Test User",
        "is_active": True,
        "is_admin": False,
    }


@pytest.fixture
def sample_session():
    """
    Create a sample session dictionary for testing.
    """
    return {
        "title": "Test Session",
        "memory_type": "sliding_window",
        "context_window": 10,
    }


@pytest.fixture
def sample_message():
    """
    Create a sample message dictionary for testing.
    """
    return {
        "role": "user",
        "content": "Hello, how are you?",
        "intent": "chitchat",
        "status": "completed",
    }


@pytest.fixture
async def app_client(db_session):
    """
    Create an async HTTP client for testing FastAPI endpoints.

    This fixture uses the real FastAPI app with a test database.
    """
    # Import here to avoid circular imports
    from unittest.mock import AsyncMock, patch

    # Create app
    app = create_app()

    # Override database dependency to use test database
    async def override_get_db():
        yield db_session

    from app.api.deps import get_db
    app.dependency_overrides[get_db] = override_get_db

    # Create test client
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test"
    ) as client:
        yield client

    # Clean up
    app.dependency_overrides.clear()


@pytest.fixture
async def test_token(app_client) -> str:
    """
    Create a test authentication token.

    This fixture registers a user and returns their access token.
    """
    # Register a test user
    response = await app_client.post(
        "/api/v1/auth/register",
        json={
            "email": "test@example.com",
            "password": "TestPass123",
            "full_name": "Test User",
        }
    )

    # Login to get token
    response = await app_client.post(
        "/api/v1/auth/login",
        json={
            "email": "test@example.com",
            "password": "TestPass123",
        }
    )

    return response.json()["access_token"]
