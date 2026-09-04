"""Tests for authentication endpoints"""
import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI


class TestAuthEndpoints:
    """Test authentication API endpoints"""

    @pytest.mark.asyncio
    async def test_register_user_success(self, app_client: AsyncClient):
        """Test successful user registration"""
        # Arrange
        response = await app_client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "SecurePass123",
                "full_name": "New User",
            },
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["email"] == "newuser@example.com"
        assert data["full_name"] == "New User"
        assert data["is_active"] is True
        assert "hashed_password" not in data  # Password should not be in response

    @pytest.mark.asyncio
    async def test_register_user_duplicate_email(self, app_client: AsyncClient):
        """Test registration with duplicate email"""
        # Arrange - Register first user
        await app_client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "SecurePass123",
            },
        )

        # Act - Try to register with same email
        response = await app_client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "DifferentPass123",
            },
        )

        # Assert
        assert response.status_code == 409  # Conflict
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_register_user_weak_password(self, app_client: AsyncClient):
        """Test registration with weak password"""
        # Act
        response = await app_client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "weak",
            },
        )

        # Assert
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_register_user_invalid_email(self, app_client: AsyncClient):
        """Test registration with invalid email"""
        # Act
        response = await app_client.post(
            "/api/v1/auth/register",
            json={
                "email": "not-an-email",
                "password": "SecurePass123",
            },
        )

        # Assert
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_login_user_success(self, app_client: AsyncClient):
        """Test successful user login"""
        # Arrange - Register user first
        await app_client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "SecurePass123",
            },
        )

        # Act
        response = await app_client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "SecurePass123",
            },
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
        assert "refresh_token" in data

    @pytest.mark.asyncio
    async def test_login_user_wrong_password(self, app_client: AsyncClient):
        """Test login with wrong password"""
        # Arrange
        await app_client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "SecurePass123",
            },
        )

        # Act
        response = await app_client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "WrongPassword123",
            },
        )

        # Assert
        assert response.status_code == 401  # Unauthorized
        data = response.json()
        assert "detail" in data

    @pytest.mark.asyncio
    async def test_login_user_not_found(self, app_client: AsyncClient):
        """Test login with non-existent user"""
        # Act
        response = await app_client.post(
            "/api/v1/auth/login",
            json={
                "email": "nonexistent@example.com",
                "password": "AnyPass123",
            },
        )

        # Assert
        assert response.status_code == 401  # Unauthorized

    @pytest.mark.asyncio
    async def test_refresh_token_success(self, app_client: AsyncClient):
        """Test refreshing access token"""
        # Arrange - Register and login to get refresh token
        await app_client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "SecurePass123",
            },
        )

        login_response = await app_client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "SecurePass123",
            },
        )
        refresh_token = login_response.json()["refresh_token"]

        # Act
        response = await app_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": refresh_token},
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_refresh_token_invalid(self, app_client: AsyncClient):
        """Test refreshing with invalid token"""
        # Act
        response = await app_client.post(
            "/api/v1/auth/refresh",
            json={"refresh_token": "invalid.token.string"},
        )

        # Assert
        assert response.status_code == 401  # Unauthorized

    @pytest.mark.asyncio
    async def test_get_current_user(self, app_client: AsyncClient):
        """Test getting current user info"""
        # Arrange - Register and login
        await app_client.post(
            "/api/v1/auth/register",
            json={
                "email": "test@example.com",
                "password": "SecurePass123",
                "full_name": "Test User",
            },
        )

        login_response = await app_client.post(
            "/api/v1/auth/login",
            json={
                "email": "test@example.com",
                "password": "SecurePass123",
            },
        )
        access_token = login_response.json()["access_token"]

        # Act
        response = await app_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["full_name"] == "Test User"
        assert "hashed_password" not in data

    @pytest.mark.asyncio
    async def test_get_current_user_no_token(self, app_client: AsyncClient):
        """Test getting current user without token"""
        # Act
        response = await app_client.get("/api/v1/auth/me")

        # Assert
        assert response.status_code == 401  # Unauthorized

    @pytest.mark.asyncio
    async def test_get_current_user_invalid_token(self, app_client: AsyncClient):
        """Test getting current user with invalid token"""
        # Act
        response = await app_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid.token.string"},
        )

        # Assert
        assert response.status_code == 401  # Unauthorized
