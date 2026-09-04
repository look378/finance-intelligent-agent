"""Tests for User repository"""
import pytest
from sqlalchemy.exc import IntegrityError


class TestUserRepository:
    """Test User repository operations"""

    @pytest.mark.asyncio
    async def test_create_user(self, db_session):
        """Test creating a new user"""
        # Arrange
        from app.repositories.user_repository import UserRepository
        from app.models.database.user import User

        user_data = User(
            email="test@example.com",
            hashed_password="hashed_password_here",
            full_name="Test User",
            is_active=True,
        )
        repo = UserRepository(db_session)

        # Act
        created_user = await repo.create(user_data)

        # Assert
        assert created_user.id is not None
        assert created_user.email == "test@example.com"
        assert created_user.full_name == "Test User"
        assert created_user.is_active is True

    @pytest.mark.asyncio
    async def test_get_user_by_id(self, db_session):
        """Test getting a user by ID"""
        # Arrange
        from app.repositories.user_repository import UserRepository
        from app.models.database.user import User

        user = User(
            email="test@example.com",
            hashed_password="hash",
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        repo = UserRepository(db_session)

        # Act
        found_user = await repo.get_by_id(user.id)

        # Assert
        assert found_user is not None
        assert found_user.id == user.id
        assert found_user.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_get_user_by_id_not_found(self, db_session):
        """Test getting a non-existent user by ID"""
        # Arrange
        from app.repositories.user_repository import UserRepository

        repo = UserRepository(db_session)

        # Act
        found_user = await repo.get_by_id(99999)

        # Assert
        assert found_user is None

    @pytest.mark.asyncio
    async def test_get_user_by_email(self, db_session):
        """Test getting a user by email"""
        # Arrange
        from app.repositories.user_repository import UserRepository
        from app.models.database.user import User

        user = User(
            email="test@example.com",
            hashed_password="hash",
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        repo = UserRepository(db_session)

        # Act
        found_user = await repo.get_by_email("test@example.com")

        # Assert
        assert found_user is not None
        assert found_user.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_get_user_by_email_not_found(self, db_session):
        """Test getting a non-existent user by email"""
        # Arrange
        from app.repositories.user_repository import UserRepository

        repo = UserRepository(db_session)

        # Act
        found_user = await repo.get_by_email("nonexistent@example.com")

        # Assert
        assert found_user is None

    @pytest.mark.asyncio
    async def test_get_user_by_email_case_insensitive(self, db_session):
        """Test that email lookup is case-insensitive"""
        # Arrange
        from app.repositories.user_repository import UserRepository
        from app.models.database.user import User

        user = User(
            email="Test@Example.com",
            hashed_password="hash",
        )
        db_session.add(user)
        await db_session.commit()

        repo = UserRepository(db_session)

        # Act
        found_user = await repo.get_by_email("test@example.com")

        # Assert
        assert found_user is not None
        assert found_user.email == "Test@Example.com"

    @pytest.mark.asyncio
    async def test_update_user(self, db_session):
        """Test updating a user"""
        # Arrange
        from app.repositories.user_repository import UserRepository
        from app.models.database.user import User

        user = User(
            email="test@example.com",
            hashed_password="hash",
            full_name="Old Name",
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        repo = UserRepository(db_session)

        # Act
        user.full_name = "New Name"
        updated_user = await repo.update(user)

        # Assert
        assert updated_user.full_name == "New Name"

    @pytest.mark.asyncio
    async def test_delete_user(self, db_session):
        """Test deleting a user"""
        # Arrange
        from app.repositories.user_repository import UserRepository
        from app.models.database.user import User

        user = User(
            email="test@example.com",
            hashed_password="hash",
        )
        db_session.add(user)
        await db_session.commit()
        await db_session.refresh(user)

        repo = UserRepository(db_session)

        # Act
        await repo.delete(user)

        # Assert
        found_user = await repo.get_by_id(user.id)
        assert found_user is None

    @pytest.mark.asyncio
    async def test_user_exists_by_email(self, db_session):
        """Test checking if user exists by email"""
        # Arrange
        from app.repositories.user_repository import UserRepository
        from app.models.database.user import User

        user = User(
            email="test@example.com",
            hashed_password="hash",
        )
        db_session.add(user)
        await db_session.commit()

        repo = UserRepository(db_session)

        # Act
        exists = await repo.exists_by_email("test@example.com")

        # Assert
        assert exists is True

    @pytest.mark.asyncio
    async def test_user_not_exists_by_email(self, db_session):
        """Test checking if non-existent user exists by email"""
        # Arrange
        from app.repositories.user_repository import UserRepository

        repo = UserRepository(db_session)

        # Act
        exists = await repo.exists_by_email("nonexistent@example.com")

        # Assert
        assert exists is False

    @pytest.mark.asyncio
    async def test_list_users_with_pagination(self, db_session):
        """Test listing users with pagination"""
        # Arrange
        from app.repositories.user_repository import UserRepository
        from app.models.database.user import User

        for i in range(5):
            user = User(
                email=f"user{i}@example.com",
                hashed_password="hash",
            )
            db_session.add(user)
        await db_session.commit()

        repo = UserRepository(db_session)

        # Act - Get first 3 users
        users = await repo.list_users(skip=0, limit=3)

        # Assert
        assert len(users) == 3

        # Act - Get next 2 users
        users = await repo.list_users(skip=3, limit=3)

        # Assert
        assert len(users) == 2
