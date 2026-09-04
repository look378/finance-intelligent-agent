"""Tests for base repository"""
import pytest
from unittest.mock import Mock, AsyncMock, patch


class TestBaseRepository:
    """Test base repository functionality"""

    def test_base_repository_initialization(self):
        """Test that base repository can be initialized"""
        # Arrange & Act
        from app.repositories.base import BaseRepository
        mock_session = Mock()

        repo = BaseRepository(mock_session)

        # Assert
        assert repo.session is mock_session

    @pytest.mark.asyncio
    async def test_get_by_id(self):
        """Test getting entity by ID"""
        # Arrange
        from app.repositories.base import BaseRepository
        from app.models.database.user import User

        mock_session = Mock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = User(
            id=1,
            email="test@example.com",
            hashed_password="hash",
        )
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = BaseRepository(mock_session)

        # Act
        result = await repo.get_by_id(1, User)

        # Assert
        assert result is not None
        assert result.id == 1
        assert result.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self):
        """Test getting entity by ID when not found"""
        # Arrange
        from app.repositories.base import BaseRepository
        from app.models.database.user import User

        mock_session = Mock()
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = BaseRepository(mock_session)

        # Act
        result = await repo.get_by_id(999, User)

        # Assert
        assert result is None

    @pytest.mark.asyncio
    async def test_get_all(self):
        """Test getting all entities"""
        # Arrange
        from app.repositories.base import BaseRepository
        from app.models.database.user import User

        mock_session = Mock()
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = [
            User(id=1, email="user1@example.com", hashed_password="hash"),
            User(id=2, email="user2@example.com", hashed_password="hash"),
        ]
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = BaseRepository(mock_session)

        # Act
        results = await repo.get_all(User)

        # Assert
        assert len(results) == 2
        assert results[0].email == "user1@example.com"
        assert results[1].email == "user2@example.com"

    @pytest.mark.asyncio
    async def test_create(self):
        """Test creating a new entity"""
        # Arrange
        from app.repositories.base import BaseRepository
        from app.models.database.user import User

        mock_session = Mock()
        mock_session.add = Mock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        new_user = User(
            email="new@example.com",
            hashed_password="hash",
        )
        mock_session.refresh.side_effect = lambda obj: setattr(obj, 'id', 1)

        repo = BaseRepository(mock_session)

        # Act
        result = await repo.create(new_user)

        # Assert
        assert result.id == 1
        mock_session.add.assert_called_once_with(new_user)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update(self):
        """Test updating an entity"""
        # Arrange
        from app.repositories.base import BaseRepository
        from app.models.database.user import User

        mock_session = Mock()
        mock_session.commit = AsyncMock()
        mock_session.refresh = AsyncMock()

        existing_user = User(
            id=1,
            email="old@example.com",
            hashed_password="hash",
        )

        repo = BaseRepository(mock_session)

        # Act
        existing_user.email = "new@example.com"
        result = await repo.update(existing_user)

        # Assert
        assert result.email == "new@example.com"
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test deleting an entity"""
        # Arrange
        from app.repositories.base import BaseRepository
        from app.models.database.user import User

        mock_session = Mock()
        mock_session.delete = Mock()
        mock_session.commit = AsyncMock()

        user_to_delete = User(
            id=1,
            email="delete@example.com",
            hashed_password="hash",
        )

        repo = BaseRepository(mock_session)

        # Act
        await repo.delete(user_to_delete)

        # Assert
        mock_session.delete.assert_called_once_with(user_to_delete)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_count(self):
        """Test counting entities"""
        # Arrange
        from app.repositories.base import BaseRepository
        from app.models.database.user import User

        mock_session = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = 42
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = BaseRepository(mock_session)

        # Act
        count = await repo.count(User)

        # Assert
        assert count == 42

    @pytest.mark.asyncio
    async def test_exists_true(self):
        """Test checking if entity exists (returns True)"""
        # Arrange
        from app.repositories.base import BaseRepository
        from app.models.database.user import User

        mock_session = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = BaseRepository(mock_session)

        # Act
        exists = await repo.exists(User.id == 1)

        # Assert
        assert exists is True

    @pytest.mark.asyncio
    async def test_exists_false(self):
        """Test checking if entity exists (returns False)"""
        # Arrange
        from app.repositories.base import BaseRepository
        from app.models.database.user import User

        mock_session = Mock()
        mock_result = Mock()
        mock_result.scalar.return_value = None
        mock_session.execute = AsyncMock(return_value=mock_result)

        repo = BaseRepository(mock_session)

        # Act
        exists = await repo.exists(User.id == 999)

        # Assert
        assert exists is False
