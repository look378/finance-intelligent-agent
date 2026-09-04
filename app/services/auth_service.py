"""
Authentication service for user authentication and authorization.

Handles user registration, login, token creation, and token verification.
"""
from typing import Dict, Any

from app.repositories.user_repository import UserRepository
from app.models.database.user import User
from app.core.security import (
    verify_password,
    hash_password,
    validate_password,
    create_access_token as create_jwt_access_token,
    create_refresh_token as create_jwt_refresh_token,
    decode_access_token as decode_jwt_token,
)
from app.core.exceptions import (
    ValidationError,
    AuthenticationError,
    ConflictError,
)


class AuthenticationService:
    """
    Service for handling authentication operations.

    Provides methods for user registration, authentication,
    and token management.
    """

    def __init__(self, user_repository: UserRepository) -> None:
        """
        Initialize the authentication service.

        Args:
            user_repository: User repository instance
        """
        self.user_repository = user_repository

    async def register_user(
        self,
        email: str,
        password: str,
        full_name: str | None = None,
    ) -> User:
        """
        Register a new user.

        Args:
            email: User email address
            password: Plain text password
            full_name: User's full name (optional)

        Returns:
            User: Created user instance

        Raises:
            ValidationError: If password validation fails
            ConflictError: If user with email already exists
        """
        # Validate password strength
        password_validation = validate_password(password)
        if not password_validation["is_valid"]:
            raise ValidationError(
                f"Password validation failed: {', '.join(password_validation['errors'])}"
            )

        # Check if user already exists
        existing_user = await self.user_repository.get_by_email(email)
        if existing_user:
            raise ConflictError(f"User with email '{email}' already exists")

        # Hash password
        hashed_password = hash_password(password)

        # Create user
        user = User(
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
            is_active=True,
            is_admin=False,
        )

        created_user = await self.user_repository.create(user)
        return created_user

    async def authenticate_user(
        self,
        email: str,
        password: str,
    ) -> User:
        """
        Authenticate a user with email and password.

        Args:
            email: User email address
            password: Plain text password

        Returns:
            User: Authenticated user instance

        Raises:
            AuthenticationError: If authentication fails
        """
        # Get user by email
        user = await self.user_repository.get_by_email(email)

        if not user:
            raise AuthenticationError("Invalid email or password")

        # Check if user is active
        if not user.is_active:
            raise AuthenticationError("User account is inactive")

        # Verify password
        if not verify_password(password, user.hashed_password):
            raise AuthenticationError("Invalid email or password")

        return user

    async def create_access_token(self, user: User) -> Dict[str, str | int]:
        """
        Create an access token for a user.

        Args:
            user: User instance

        Returns:
            dict: Dictionary containing access_token, token_type, and expires_in
        """
        from app.config.settings import settings

        token_data = {
            "sub": user.email,
            "user_id": user.id,
        }

        access_token = create_jwt_access_token(token_data)

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,  # Convert to seconds
        }

    async def create_refresh_token(self, user: User) -> str:
        """
        Create a refresh token for a user.

        Args:
            user: User instance

        Returns:
            str: Refresh token
        """
        token_data = {
            "sub": user.email,
            "user_id": user.id,
        }

        refresh_token = create_jwt_refresh_token(token_data)
        return refresh_token

    async def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode an access token.

        Args:
            token: JWT access token

        Returns:
            dict: Decoded token payload

        Raises:
            AuthenticationError: If token is invalid or expired
        """
        try:
            payload = decode_jwt_token(token)

            # Verify token type
            if payload.get("type") != "access":
                raise AuthenticationError("Invalid token type")

            return payload

        except Exception as e:
            raise AuthenticationError(f"Token verification failed: {str(e)}") from e

    async def refresh_access_token(self, refresh_token: str) -> Dict[str, str | int]:
        """
        Refresh an access token using a refresh token.

        Args:
            refresh_token: JWT refresh token

        Returns:
            dict: Dictionary containing new access_token, token_type, and expires_in

        Raises:
            AuthenticationError: If refresh token is invalid or expired
        """
        try:
            payload = decode_jwt_token(refresh_token)

            # Verify token type
            if payload.get("type") != "refresh":
                raise AuthenticationError("Invalid refresh token")

            # Get user from token
            user_id = payload.get("user_id")
            if not user_id:
                raise AuthenticationError("Invalid token payload")

            user = await self.user_repository.get_by_id(user_id, User)
            if not user:
                raise AuthenticationError("User not found")

            if not user.is_active:
                raise AuthenticationError("User account is inactive")

            # Create new access token
            return await self.create_access_token(user)

        except Exception as e:
            raise AuthenticationError(f"Token refresh failed: {str(e)}") from e
