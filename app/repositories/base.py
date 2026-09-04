"""
Base repository with common CRUD operations.

Provides generic database operations using SQLAlchemy async.
All repositories should inherit from this base class.
"""
from typing import TypeVar, Type, Generic, List, Dict, Any

from pydantic import BaseModel
from sqlalchemy import select, delete, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeMeta

from app.models.database.base import Base

ModelType = TypeVar("ModelType", bound=Base)
CreateSchemaType = TypeVar("CreateSchemaType", bound=BaseModel)
UpdateSchemaType = TypeVar("UpdateSchemaType", bound=BaseModel)


class BaseRepository(Generic[ModelType]):
    """
    Base repository with generic CRUD operations.

    This class provides common database operations that can be used
    by all specific repositories. It uses SQLAlchemy async for
    asynchronous database access.

    Type Parameters:
        ModelType: The SQLAlchemy model type

    Attributes:
        session: The async database session
    """

    def __init__(self, session: AsyncSession) -> None:
        """
        Initialize the repository.

        Args:
            session: Async database session
        """
        self.session = session

    async def get_by_id(
        self,
        id: int,
        model: Type[ModelType],
    ) -> ModelType | None:
        """
        Get an entity by its primary key ID.

        Args:
            id: Primary key ID
            model: SQLAlchemy model class

        Returns:
            ModelType | None: The entity if found, None otherwise
        """
        stmt = select(model).where(model.id == id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all(
        self,
        model: Type[ModelType],
        skip: int = 0,
        limit: int = 100,
    ) -> List[ModelType]:
        """
        Get all entities with pagination.

        Args:
            model: SQLAlchemy model class
            skip: Number of records to skip (for pagination)
            limit: Maximum number of records to return

        Returns:
            List[ModelType]: List of entities
        """
        stmt = select(model).offset(skip).limit(limit)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def create(
        self,
        entity: ModelType,
    ) -> ModelType:
        """
        Create a new entity.

        Args:
            entity: Entity instance to create

        Returns:
            ModelType: Created entity with ID assigned
        """
        self.session.add(entity)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity

    async def update(
        self,
        entity: ModelType,
    ) -> ModelType:
        """
        Update an existing entity.

        Note: The entity should be modified before calling this method.
        This method simply commits the changes.

        Args:
            entity: Entity instance with modified fields

        Returns:
            ModelType: Updated entity
        """
        await self.session.commit()
        await self.session.refresh(entity)
        return entity

    async def delete(
        self,
        entity: ModelType,
    ) -> None:
        """
        Delete an entity.

        Args:
            entity: Entity instance to delete
        """
        await self.session.delete(entity)
        await self.session.commit()

    async def count(
        self,
        model: Type[ModelType],
    ) -> int:
        """
        Count all entities of a given model.

        Args:
            model: SQLAlchemy model class

        Returns:
            int: Number of entities
        """
        stmt = select(func.count()).select_from(model)
        result = await self.session.execute(stmt)
        return result.scalar() or 0

    async def exists(
        self,
        condition,
    ) -> bool:
        """
        Check if any entity matching the condition exists.

        Args:
            condition: SQLAlchemy condition (e.g., Model.id == 1)

        Returns:
            bool: True if any matching entity exists, False otherwise
        """
        stmt = select(func.count()).where(condition)
        result = await self.session.execute(stmt)
        count = result.scalar() or 0
        return count > 0
