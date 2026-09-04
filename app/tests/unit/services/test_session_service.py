"""Tests for Session service"""
import pytest
from unittest.mock import Mock, AsyncMock


class TestSessionService:
    """Test Session service"""

    @pytest.mark.asyncio
    async def test_create_session(self, db_session):
        """Test creating a new session"""
        # Arrange
        from app.services.session_service import SessionService
        from app.repositories.session_repository import SessionRepository
        from app.repositories.user_repository import UserRepository
        from app.models.database.user import User

        user = User(
            email="test@example.com",
            hashed_password="hash",
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        session_repo = SessionRepository(db_session)
        session_service = SessionService(session_repo)

        # Act
        session = await session_service.create_session(
            user_id=user.id,
            title="My Chat Session",
            memory_type="sliding_window",
            context_window=10,
        )

        # Assert
        assert session is not None
        assert session.id is not None
        assert session.user_id == user.id
        assert session.title == "My Chat Session"
        assert session.memory_type == "sliding_window"
        assert session.context_window == 10

    @pytest.mark.asyncio
    async def test_create_session_default_values(self, db_session):
        """Test creating session with default values"""
        # Arrange
        from app.services.session_service import SessionService
        from app.repositories.session_repository import SessionRepository
        from app.repositories.user_repository import UserRepository
        from app.models.database.user import User

        user = User(
            email="test@example.com",
            hashed_password="hash",
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        session_repo = SessionRepository(db_session)
        session_service = SessionService(session_repo)

        # Act
        session = await session_service.create_session(user_id=user.id)

        # Assert
        assert session.title == "New Chat"
        assert session.memory_type == "sliding_window"
        assert session.context_window == 10

    @pytest.mark.asyncio
    async def test_get_user_sessions(self, db_session):
        """Test getting all sessions for a user"""
        # Arrange
        from app.services.session_service import SessionService
        from app.repositories.session_repository import SessionRepository
        from app.repositories.user_repository import UserRepository
        from app.models.database.user import User
        from app.models.database.session import ChatSession

        user = User(
            email="test@example.com",
            hashed_password="hash",
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        # Create multiple sessions
        for i in range(3):
            session = ChatSession(
                user_id=user.id,
                title=f"Session {i}",
            )
            db_session.add(session)
        await db_session.commit()

        session_repo = SessionRepository(db_session)
        session_service = SessionService(session_repo)

        # Act
        sessions = await session_service.get_user_sessions(user.id)

        # Assert
        assert len(sessions) == 3
        assert all(s.user_id == user.id for s in sessions)

    @pytest.mark.asyncio
    async def test_get_session_by_id(self, db_session):
        """Test getting a session by ID"""
        # Arrange
        from app.services.session_service import SessionService
        from app.repositories.session_repository import SessionRepository
        from app.repositories.user_repository import UserRepository
        from app.models.database.user import User
        from app.models.database.session import ChatSession
        from app.core.exceptions import NotFoundError

        user = User(
            email="test@example.com",
            hashed_password="hash",
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        session = ChatSession(
            user_id=user.id,
            title="Test Session",
        )
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)

        session_repo = SessionRepository(db_session)
        session_service = SessionService(session_repo)

        # Act
        found_session = await session_service.get_session_by_id(session.id, user.id)

        # Assert
        assert found_session is not None
        assert found_session.id == session.id
        assert found_session.user_id == user.id

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, db_session):
        """Test getting a non-existent session"""
        # Arrange
        from app.services.session_service import SessionService
        from app.repositories.session_repository import SessionRepository
        from app.core.exceptions import NotFoundError

        session_repo = SessionRepository(db_session)
        session_service = SessionService(session_repo)

        # Act & Assert
        with pytest.raises(NotFoundError):
            await session_service.get_session_by_id(99999, 1)

    @pytest.mark.asyncio
    async def test_get_session_unauthorized_user(self, db_session):
        """Test getting a session that belongs to another user"""
        # Arrange
        from app.services.session_service import SessionService
        from app.repositories.session_repository import SessionRepository
        from app.repositories.user_repository import UserRepository
        from app.models.database.user import User
        from app.models.database.session import ChatSession
        from app.core.exceptions import NotFoundError

        # Create two users
        user1 = User(email="user1@example.com", hashed_password="hash")
        user2 = User(email="user2@example.com", hashed_password="hash")
        db_session.add(user1)
        db_session.add(user2)
        await db_session.commit()
        await db_session.refresh(user1)
        await db_session.refresh(user2)

        # Create session for user1
        session = ChatSession(
            user_id=user1.id,
            title="User1's Session",
        )
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)

        session_repo = SessionRepository(db_session)
        session_service = SessionService(session_repo)

        # Act & Assert - User2 tries to access User1's session
        with pytest.raises(NotFoundError):
            await session_service.get_session_by_id(session.id, user2.id)

    @pytest.mark.asyncio
    async def test_update_session(self, db_session):
        """Test updating a session"""
        # Arrange
        from app.services.session_service import SessionService
        from app.repositories.session_repository import SessionRepository
        from app.repositories.user_repository import UserRepository
        from app.models.database.user import User
        from app.models.database.session import ChatSession

        user = User(
            email="test@example.com",
            hashed_password="hash",
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        session = ChatSession(
            user_id=user.id,
            title="Old Title",
            memory_type="sliding_window",
        )
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)

        session_repo = SessionRepository(db_session)
        session_service = SessionService(session_repo)

        # Act
        updated_session = await session_service.update_session(
            session_id=session.id,
            user_id=user.id,
            title="New Title",
            memory_type="summarization",
            context_window=20,
        )

        # Assert
        assert updated_session.title == "New Title"
        assert updated_session.memory_type == "summarization"
        assert updated_session.context_window == 20

    @pytest.mark.asyncio
    async def test_delete_session(self, db_session):
        """Test deleting a session"""
        # Arrange
        from app.services.session_service import SessionService
        from app.repositories.session_repository import SessionRepository
        from app.repositories.user_repository import UserRepository
        from app.models.database.user import User
        from app.models.database.session import ChatSession

        user = User(
            email="test@example.com",
            hashed_password="hash",
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        session = ChatSession(
            user_id=user.id,
            title="To Delete",
        )
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)

        session_repo = SessionRepository(db_session)
        session_service = SessionService(session_repo)

        # Act
        await session_service.delete_session(session.id, user.id)

        # Assert
        found_session = await session_repo.get_by_id(session.id)
        assert found_session is None
