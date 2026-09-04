"""Tests for Authentication service"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import timedelta


class TestAuthenticationService:
    """Test authentication service"""

    @pytest.mark.asyncio
    async def test_register_user_success(self, db_session):
        """Test successful user registration"""
        # Arrange
        from app.services.auth_service import AuthenticationService
        from app.repositories.user_repository import UserRepository

        user_repo = UserRepository(db_session)
        auth_service = AuthenticationService(user_repo)

        user_data = {
            "email": "newuser@example.com",
            "password": "SecurePass123",
            "full_name": "New User",
        }

        # Act
        user = await auth_service.register_user(**user_data)

        # Assert
        assert user is not None
        assert user.id is not None
        assert user.email == "newuser@example.com"
        assert user.full_name == "New User"
        assert user.is_active is True
        assert user.hashed_password != "SecurePass123"  # Password should be hashed

    @pytest.mark.asyncio
    async def test_register_user_duplicate_email(self, db_session):
        """Test registering with duplicate email raises error"""
        # Arrange
        from app.services.auth_service import AuthenticationService
        from app.repositories.user_repository import UserRepository
        from app.models.database.user import User
        from app.core.exceptions import ConflictError

        user_repo = UserRepository(db_session)
        auth_service = AuthenticationService(user_repo)

        # Create existing user
        existing_user = User(
            email="existing@example.com",
            hashed_password="hash",
        )
        db_session.add(existing_user)
        await db_session.commit()

        # Act & Assert
        with pytest.raises(ConflictError) as exc_info:
            await auth_service.register_user(
                email="existing@example.com",
                password="SecurePass123",
            )
        assert "already exists" in str(exc_info.value).lower() or "email" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_register_user_weak_password(self, db_session):
        """Test registering with weak password raises error"""
        # Arrange
        from app.services.auth_service import AuthenticationService
        from app.repositories.user_repository import UserRepository
        from app.core.exceptions import ValidationError

        user_repo = UserRepository(db_session)
        auth_service = AuthenticationService(user_repo)

        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            await auth_service.register_user(
                email="test@example.com",
                password="weak",
            )
        assert "password" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_authenticate_user_success(self, db_session):
        """Test successful authentication"""
        # Arrange
        from app.services.auth_service import AuthenticationService
        from app.repositories.user_repository import UserRepository
        from app.core.security import hash_password
        from app.models.database.user import User

        # Create user with known password
        password = "CorrectPass123"
        user = User(
            email="test@example.com",
            hashed_password=hash_password(password),
            is_active=True,
        )
        db_session.add(user)
        await db_session.commit()

        user_repo = UserRepository(db_session)
        auth_service = AuthenticationService(user_repo)

        # Act
        authenticated_user = await auth_service.authenticate_user(
            email="test@example.com",
            password=password,
        )

        # Assert
        assert authenticated_user is not None
        assert authenticated_user.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_authenticate_user_wrong_password(self, db_session):
        """Test authentication with wrong password"""
        # Arrange
        from app.services.auth_service import AuthenticationService
        from app.repositories.user_repository import UserRepository
        from app.core.security import hash_password
        from app.models.database.user import User
        from app.core.exceptions import AuthenticationError

        password = "CorrectPass123"
        user = User(
            email="test@example.com",
            hashed_password=hash_password(password),
            is_active=True,
        )
        db_session.add(user)
        await db_session.commit()

        user_repo = UserRepository(db_session)
        auth_service = AuthenticationService(user_repo)

        # Act & Assert
        with pytest.raises(AuthenticationError):
            await auth_service.authenticate_user(
                email="test@example.com",
                password="WrongPass123",
            )

    @pytest.mark.asyncio
    async def test_authenticate_user_not_found(self, db_session):
        """Test authentication with non-existent user"""
        # Arrange
        from app.services.auth_service import AuthenticationService
        from app.repositories.user_repository import UserRepository
        from app.core.exceptions import AuthenticationError

        user_repo = UserRepository(db_session)
        auth_service = AuthenticationService(user_repo)

        # Act & Assert
        with pytest.raises(AuthenticationError):
            await auth_service.authenticate_user(
                email="nonexistent@example.com",
                password="AnyPass123",
            )

    @pytest.mark.asyncio
    async def test_authenticate_user_inactive(self, db_session):
        """Test authentication with inactive user"""
        # Arrange
        from app.services.auth_service import AuthenticationService
        from app.repositories.user_repository import UserRepository
        from app.core.security import hash_password
        from app.models.database.user import User
        from app.core.exceptions import AuthenticationError

        password = "AnyPass123"
        user = User(
            email="test@example.com",
            hashed_password=hash_password(password),
            is_active=False,  # Inactive user
        )
        db_session.add(user)
        await db_session.commit()

        user_repo = UserRepository(db_session)
        auth_service = AuthenticationService(user_repo)

        # Act & Assert
        with pytest.raises(AuthenticationError) as exc_info:
            await auth_service.authenticate_user(
                email="test@example.com",
                password=password,
            )
        assert "inactive" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_create_access_token(self, db_session):
        """Test creating access token for user"""
        # Arrange
        from app.services.auth_service import AuthenticationService
        from app.repositories.user_repository import UserRepository
        from app.models.database.user import User

        user = User(
            id=1,
            email="test@example.com",
            hashed_password="hash",
        )

        user_repo = UserRepository(db_session)
        auth_service = AuthenticationService(user_repo)

        # Act
        token_data = await auth_service.create_access_token(user)

        # Assert
        assert "access_token" in token_data
        assert "token_type" in token_data
        assert "expires_in" in token_data
        assert token_data["token_type"] == "bearer"
        assert isinstance(token_data["access_token"], str)
        assert len(token_data["access_token"]) > 0

    @pytest.mark.asyncio
    async def test_create_refresh_token(self, db_session):
        """Test creating refresh token for user"""
        # Arrange
        from app.services.auth_service import AuthenticationService
        from app.repositories.user_repository import UserRepository
        from app.models.database.user import User

        user = User(
            id=1,
            email="test@example.com",
            hashed_password="hash",
        )

        user_repo = UserRepository(db_session)
        auth_service = AuthenticationService(user_repo)

        # Act
        token = await auth_service.create_refresh_token(user)

        # Assert
        assert isinstance(token, str)
        assert len(token) > 0

    @pytest.mark.asyncio
    async def test_verify_token_valid(self, db_session):
        """Test verifying a valid token"""
        # Arrange
        from app.services.auth_service import AuthenticationService
        from app.repositories.user_repository import UserRepository
        from app.models.database.user import User

        user = User(
            id=123,
            email="test@example.com",
            hashed_password="hash",
        )

        user_repo = UserRepository(db_session)
        auth_service = AuthenticationService(user_repo)

        token_data = await auth_service.create_access_token(user)
        token = token_data["access_token"]

        # Act
        payload = await auth_service.verify_token(token)

        # Assert
        assert payload is not None
        assert payload["sub"] == "test@example.com"
        assert "user_id" in payload
        assert payload["user_id"] == 123

    @pytest.mark.asyncio
    async def test_verify_token_invalid(self, db_session):
        """Test verifying an invalid token"""
        # Arrange
        from app.services.auth_service import AuthenticationService
        from app.repositories.user_repository import UserRepository
        from app.core.exceptions import AuthenticationError

        user_repo = UserRepository(db_session)
        auth_service = AuthenticationService(user_repo)

        # Act & Assert
        with pytest.raises(AuthenticationError):
            await auth_service.verify_token("invalid.token.string")

    @pytest.mark.asyncio
    async def test_refresh_access_token(self, db_session):
        """Test refreshing access token with refresh token"""
        # Arrange
        from app.services.auth_service import AuthenticationService
        from app.repositories.user_repository import UserRepository
        from app.models.database.user import User

        user = User(
            id=123,
            email="test@example.com",
            hashed_password="hash",
        )

        user_repo = UserRepository(db_session)
        auth_service = AuthenticationService(user_repo)

        refresh_token = await auth_service.create_refresh_token(user)

        # Act
        new_token_data = await auth_service.refresh_access_token(refresh_token)

        # Assert
        assert "access_token" in new_token_data
        assert new_token_data["token_type"] == "bearer"
