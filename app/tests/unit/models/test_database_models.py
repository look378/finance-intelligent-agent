"""Tests for database models"""
import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session


class TestBaseModel:
    """Test base database model"""

    def test_base_model_has_timestamps(self, db_session):
        """Test that base model includes created_at and updated_at"""
        # Arrange & Act
        from app.models.database.base import Base, TimestampMixin

        # Assert - TimestampMixin should have these attributes
        assert hasattr(TimestampMixin, 'created_at')
        assert hasattr(TimestampMixin, 'updated_at')

    def test_timestamp_defaults(self, db_session):
        """Test that timestamps are set automatically"""
        # This will be tested with concrete models
        pass


class TestUserModel:
    """Test User model"""

    def test_user_creation(self, db_session):
        """Test creating a user"""
        # Arrange
        from app.models.database.user import User

        # Act
        user = User(
            email="test@example.com",
            hashed_password="hashed_password_here",
            full_name="Test User",
            is_active=True,
            is_admin=False,
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

        # Assert
        assert user.id is not None
        assert user.email == "test@example.com"
        assert user.hashed_password == "hashed_password_here"
        assert user.full_name == "Test User"
        assert user.is_active is True
        assert user.is_admin is False
        assert user.created_at is not None
        assert user.updated_at is not None

    def test_user_email_unique(self, db_session):
        """Test that user emails must be unique"""
        # Arrange
        from app.models.database.user import User
        from sqlalchemy.exc import IntegrityError

        user1 = User(
            email="test@example.com",
            hashed_password="hashed_password_here",
        )
        user2 = User(
            email="test@example.com",  # Same email
            hashed_password="different_password",
        )

        # Act
        db_session.add(user1)
        db_session.add(user2)

        # Assert
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_user_email_required(self, db_session):
        """Test that user email is required"""
        # Arrange
        from app.models.database.user import User
        from sqlalchemy.exc import IntegrityError

        user = User(
            hashed_password="hashed_password_here",
        )

        # Act & Assert
        db_session.add(user)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_user_defaults(self, db_session):
        """Test that user fields have correct defaults"""
        # Arrange
        from app.models.database.user import User

        # Act
        user = User(
            email="test@example.com",
            hashed_password="hashed_password_here",
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

        # Assert
        assert user.is_active is True
        assert user.is_admin is False
        assert user.full_name is None


class TestChatSessionModel:
    """Test ChatSession model"""

    def test_session_creation(self, db_session):
        """Test creating a chat session"""
        # Arrange
        from app.models.database.user import User
        from app.models.database.session import ChatSession

        user = User(
            email="test@example.com",
            hashed_password="hashed_password_here",
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

        # Act
        session = ChatSession(
            user_id=user.id,
            title="Test Session",
            memory_type="sliding_window",
            context_window=10,
        )
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)

        # Assert
        assert session.id is not None
        assert session.user_id == user.id
        assert session.title == "Test Session"
        assert session.memory_type == "sliding_window"
        assert session.context_window == 10
        assert session.created_at is not None
        assert session.updated_at is not None

    def test_session_relationship_with_user(self, db_session):
        """Test session-user relationship"""
        # Arrange
        from app.models.database.user import User
        from app.models.database.session import ChatSession

        user = User(
            email="test@example.com",
            hashed_password="hashed_password_here",
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

        session = ChatSession(
            user_id=user.id,
            title="Test Session",
        )
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)

        # Act
        loaded_user = db_session.query(User).filter_by(id=user.id).first()

        # Assert
        assert loaded_user is not None
        # Note: relationships are tested with ORM queries

    def test_session_defaults(self, db_session):
        """Test that session fields have correct defaults"""
        # Arrange
        from app.models.database.user import User
        from app.models.database.session import ChatSession

        user = User(
            email="test@example.com",
            hashed_password="hashed_password_here",
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

        # Act
        session = ChatSession(
            user_id=user.id,
        )
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)

        # Assert
        assert session.title == "New Chat"
        assert session.memory_type == "sliding_window"
        assert session.context_window == 10


class TestMessageModel:
    """Test Message model"""

    def test_message_creation(self, db_session):
        """Test creating a message"""
        # Arrange
        from app.models.database.user import User
        from app.models.database.session import ChatSession
        from app.models.database.message import Message
        from app.models.enums.message import MessageRole, MessageStatus
        from app.models.enums.intent import Intent

        user = User(
            email="test@example.com",
            hashed_password="hashed_password_here",
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

        session = ChatSession(
            user_id=user.id,
        )
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)

        # Act
        message = Message(
            session_id=session.id,
            role=MessageRole.USER,
            content="Hello, world!",
            intent=Intent.FAQ,
            status=MessageStatus.COMPLETED,
        )
        db_session.add(message)
        db_session.commit()
        db_session.refresh(message)

        # Assert
        assert message.id is not None
        assert message.session_id == session.id
        assert message.role == MessageRole.USER
        assert message.content == "Hello, world!"
        assert message.status == MessageStatus.COMPLETED
        assert message.created_at is not None

    def test_message_defaults(self, db_session):
        """Test that message fields have correct defaults"""
        # Arrange
        from app.models.database.user import User
        from app.models.database.session import ChatSession
        from app.models.database.message import Message
        from app.models.enums.message import MessageRole, MessageStatus

        user = User(
            email="test@example.com",
            hashed_password="hashed_password_here",
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

        session = ChatSession(
            user_id=user.id,
        )
        db_session.add(session)
        db_session.commit()
        db_session.refresh(session)

        # Act
        message = Message(
            session_id=session.id,
            role=MessageRole.USER,
            content="Test message",
        )
        db_session.add(message)
        db_session.commit()
        db_session.refresh(message)

        # Assert
        assert message.status == MessageStatus.COMPLETED
        assert message.intent is None
        assert message.token_count is None

    def test_message_relationship_with_session(self, db_session):
        """Test message-session relationship"""
        # This will be tested with repository pattern
        pass


class TestDocumentModel:
    """Test Document model"""

    def test_document_creation(self, db_session):
        """Test creating a document"""
        # Arrange
        from app.models.database.document import Document

        # Act
        document = Document(
            external_doc_id="doc_123",
            title="Test Document",
            source="/path/to/doc.pdf",
            doc_type="pdf",
            chunk_count=10,
        )
        db_session.add(document)
        db_session.commit()
        db_session.refresh(document)

        # Assert
        assert document.id is not None
        assert document.external_doc_id == "doc_123"
        assert document.title == "Test Document"
        assert document.source == "/path/to/doc.pdf"
        assert document.doc_type == "pdf"
        assert document.chunk_count == 10
        assert document.is_active is True

    def test_document_external_id_unique(self, db_session):
        """Test that external_doc_id must be unique"""
        # Arrange
        from app.models.database.document import Document
        from sqlalchemy.exc import IntegrityError

        doc1 = Document(
            external_doc_id="doc_123",
            title="Document 1",
            source="source1",
            doc_type="pdf",
        )
        doc2 = Document(
            external_doc_id="doc_123",  # Same external ID
            title="Document 2",
            source="source2",
            doc_type="txt",
        )

        # Act
        db_session.add(doc1)
        db_session.add(doc2)

        # Assert
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_document_defaults(self, db_session):
        """Test that document fields have correct defaults"""
        # Arrange
        from app.models.database.document import Document

        # Act
        document = Document(
            external_doc_id="doc_123",
            title="Test Document",
            source="source",
            doc_type="pdf",
        )
        db_session.add(document)
        db_session.commit()
        db_session.refresh(document)

        # Assert
        assert document.chunk_count == 0
        assert document.is_active is True
        assert document.doc_metadata is None
