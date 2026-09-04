"""Tests for Session repository"""
import pytest


class TestSessionRepository:
    """Test Session repository operations"""

    @pytest.mark.asyncio
    async def test_create_session(self, db_session):
        """Test creating a new session"""
        # Arrange
        from app.repositories.session_repository import SessionRepository
        from app.repositories.user_repository import UserRepository
        from app.models.database.user import User
        from app.models.database.session import ChatSession

        # Create user
        user = User(
            email="test@example.com",
            hashed_password="hash",
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        session_repo = SessionRepository(db_session)

        # Act
        new_session = ChatSession(
            user_id=user.id,
            title="Test Session",
            memory_type="sliding_window",
            context_window=10,
        )
        created_session = await session_repo.create(new_session)

        # Assert
        assert created_session.id is not None
        assert created_session.user_id == user.id
        assert created_session.title == "Test Session"

    @pytest.mark.asyncio
    async def test_get_sessions_by_user(self, db_session):
        """Test getting all sessions for a user"""
        # Arrange
        from app.repositories.session_repository import SessionRepository
        from app.repositories.user_repository import UserRepository
        from app.models.database.user import User
        from app.models.database.session import ChatSession

        # Create user
        user = User(
            email="test@example.com",
            hashed_password="hash",
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        session_repo = SessionRepository(db_session)

        # Create multiple sessions
        for i in range(3):
            session = ChatSession(
                user_id=user.id,
                title=f"Session {i}",
            )
            db_session.add(session)
        await db_session.commit()

        # Act
        sessions = await session_repo.get_by_user_id(user.id)

        # Assert
        assert len(sessions) == 3
        assert all(s.user_id == user.id for s in sessions)

    @pytest.mark.asyncio
    async def test_get_session_with_messages(self, db_session):
        """Test getting a session with messages preloaded"""
        # Arrange
        from app.repositories.session_repository import SessionRepository
        from app.repositories.user_repository import UserRepository
        from app.models.database.user import User
        from app.models.database.session import ChatSession
        from app.models.database.message import Message
        from app.models.enums.message import MessageRole, MessageStatus

        # Create user and session
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

        # Add messages
        message = Message(
            session_id=session.id,
            role=MessageRole.USER,
            content="Hello",
            status=MessageStatus.COMPLETED,
        )
        db_session.add(message)
        await db_session.commit()

        session_repo = SessionRepository(db_session)

        # Act
        loaded_session = await session_repo.get_with_messages(session.id)

        # Assert
        assert loaded_session is not None
        assert loaded_session.id == session.id
        # Messages should be accessible without additional queries
        # (This depends on SQLAlchemy's relationship loading)

    @pytest.mark.asyncio
    async def test_delete_session(self, db_session):
        """Test deleting a session"""
        # Arrange
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
            title="Test Session",
        )
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)

        session_repo = SessionRepository(db_session)

        # Act
        await session_repo.delete(session)

        # Assert
        found_session = await session_repo.get_by_id(session.id)
        assert found_session is None

    @pytest.mark.asyncio
    async def test_count_user_sessions(self, db_session):
        """Test counting sessions for a user"""
        # Arrange
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

        session_repo = SessionRepository(db_session)

        # Create sessions
        for i in range(5):
            session = ChatSession(
                user_id=user.id,
                title=f"Session {i}",
            )
            db_session.add(session)
        await db_session.commit()

        # Act
        count = await session_repo.count_by_user_id(user.id)

        # Assert
        assert count == 5

    @pytest.mark.asyncio
    async def test_update_session(self, db_session):
        """Test updating a session"""
        # Arrange
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
        )
        db_session.add(session)
        await db_session.commit()
        await db_session.refresh(session)

        session_repo = SessionRepository(db_session)

        # Act
        session.title = "New Title"
        session.context_window = 20
        updated_session = await session_repo.update(session)

        # Assert
        assert updated_session.title == "New Title"
        assert updated_session.context_window == 20
