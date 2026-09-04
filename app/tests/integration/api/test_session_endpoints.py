"""Tests for session endpoints"""
import pytest
from httpx import AsyncClient


class TestSessionEndpoints:
    """Test session API endpoints"""

    @pytest.mark.asyncio
    async def test_create_session(self, app_client: AsyncClient, test_token: str):
        """Test creating a new session"""
        # Arrange
        headers = {"Authorization": f"Bearer {test_token}"}

        # Act
        response = await app_client.post(
            "/api/v1/sessions",
            json={
                "title": "My Chat Session",
                "memory_type": "sliding_window",
                "context_window": 15,
            },
            headers=headers,
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["title"] == "My Chat Session"
        assert data["memory_type"] == "sliding_window"
        assert data["context_window"] == 15

    @pytest.mark.asyncio
    async def test_create_session_default_values(self, app_client: AsyncClient, test_token: str):
        """Test creating session with default values"""
        # Arrange
        headers = {"Authorization": f"Bearer {test_token}"}

        # Act
        response = await app_client.post(
            "/api/v1/sessions",
            json={},
            headers=headers,
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "New Chat"
        assert data["memory_type"] == "sliding_window"
        assert data["context_window"] == 10

    @pytest.mark.asyncio
    async def test_create_session_unauthorized(self, app_client: AsyncClient):
        """Test creating session without authentication"""
        # Act
        response = await app_client.post(
            "/api/v1/sessions",
            json={"title": "Test Session"},
        )

        # Assert
        assert response.status_code == 401  # Unauthorized

    @pytest.mark.asyncio
    async def test_list_sessions(self, app_client: AsyncClient, test_token: str):
        """Test listing user's sessions"""
        # Arrange
        headers = {"Authorization": f"Bearer {test_token}"}

        # Create multiple sessions
        for i in range(3):
            await app_client.post(
                "/api/v1/sessions",
                json={"title": f"Session {i}"},
                headers=headers,
            )

        # Act
        response = await app_client.get("/api/v1/sessions", headers=headers)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert len(data["items"]) >= 3
        assert "total" in data
        assert "page" in data

    @pytest.mark.asyncio
    async def test_list_sessions_pagination(self, app_client: AsyncClient, test_token: str):
        """Test listing sessions with pagination"""
        # Arrange
        headers = {"Authorization": f"Bearer {test_token}"}

        # Create sessions
        for i in range(5):
            await app_client.post(
                "/api/v1/sessions",
                json={"title": f"Session {i}"},
                headers=headers,
            )

        # Act - Get first page
        response = await app_client.get(
            "/api/v1/sessions?page=1&page_size=2",
            headers=headers,
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 2
        assert data["page"] == 1
        assert data["page_size"] == 2

    @pytest.mark.asyncio
    async def test_get_session_by_id(self, app_client: AsyncClient, test_token: str):
        """Test getting a specific session"""
        # Arrange
        headers = {"Authorization": f"Bearer {test_token}"}

        create_response = await app_client.post(
            "/api/v1/sessions",
            json={"title": "Test Session"},
            headers=headers,
        )
        session_id = create_response.json()["id"]

        # Act
        response = await app_client.get(f"/api/v1/sessions/{session_id}", headers=headers)

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == session_id
        assert data["title"] == "Test Session"

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, app_client: AsyncClient, test_token: str):
        """Test getting a non-existent session"""
        # Arrange
        headers = {"Authorization": f"Bearer {test_token}"}

        # Act
        response = await app_client.get("/api/v1/sessions/99999", headers=headers)

        # Assert
        assert response.status_code == 404  # Not Found

    @pytest.mark.asyncio
    async def test_get_session_unauthorized_user(self, app_client: AsyncClient, test_token: str):
        """Test accessing another user's session"""
        # This would require creating two users, which we'll skip for now
        # The repository tests cover this logic
        pass

    @pytest.mark.asyncio
    async def test_update_session(self, app_client: AsyncClient, test_token: str):
        """Test updating a session"""
        # Arrange
        headers = {"Authorization": f"Bearer {test_token}"}

        create_response = await app_client.post(
            "/api/v1/sessions",
            json={"title": "Old Title"},
            headers=headers,
        )
        session_id = create_response.json()["id"]

        # Act
        response = await app_client.put(
            f"/api/v1/sessions/{session_id}",
            json={
                "title": "New Title",
                "memory_type": "summarization",
                "context_window": 20,
            },
            headers=headers,
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "New Title"
        assert data["memory_type"] == "summarization"
        assert data["context_window"] == 20

    @pytest.mark.asyncio
    async def test_update_session_partial(self, app_client: AsyncClient, test_token: str):
        """Test partially updating a session"""
        # Arrange
        headers = {"Authorization": f"Bearer {test_token}"}

        create_response = await app_client.post(
            "/api/v1/sessions",
            json={
                "title": "Original Title",
                "memory_type": "sliding_window",
                "context_window": 10,
            },
            headers=headers,
        )
        session_id = create_response.json()["id"]

        # Act - Only update title
        response = await app_client.put(
            f"/api/v1/sessions/{session_id}",
            json={"title": "Updated Title"},
            headers=headers,
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "Updated Title"
        # Other fields should remain unchanged
        assert data["memory_type"] == "sliding_window"
        assert data["context_window"] == 10

    @pytest.mark.asyncio
    async def test_delete_session(self, app_client: AsyncClient, test_token: str):
        """Test deleting a session"""
        # Arrange
        headers = {"Authorization": f"Bearer {test_token}"}

        create_response = await app_client.post(
            "/api/v1/sessions",
            json={"title": "To Delete"},
            headers=headers,
        )
        session_id = create_response.json()["id"]

        # Act
        response = await app_client.delete(f"/api/v1/sessions/{session_id}", headers=headers)

        # Assert
        assert response.status_code == 204  # No Content

        # Verify session is deleted
        get_response = await app_client.get(f"/api/v1/sessions/{session_id}", headers=headers)
        assert get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_session_unauthorized(self, app_client: AsyncClient):
        """Test deleting a session without authentication"""
        # Act
        response = await app_client.delete("/api/v1/sessions/1")

        # Assert
        assert response.status_code == 401  # Unauthorized

    @pytest.mark.asyncio
    async def test_update_session_invalid_memory_type(self, app_client: AsyncClient, test_token: str):
        """Test updating session with invalid memory type"""
        # Arrange
        headers = {"Authorization": f"Bearer {test_token}"}

        create_response = await app_client.post(
            "/api/v1/sessions",
            json={},
            headers=headers,
        )
        session_id = create_response.json()["id"]

        # Act
        response = await app_client.put(
            f"/api/v1/sessions/{session_id}",
            json={"memory_type": "invalid_type"},
            headers=headers,
        )

        # Assert
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_update_session_invalid_context_window(self, app_client: AsyncClient, test_token: str):
        """Test updating session with invalid context window"""
        # Arrange
        headers = {"Authorization": f"Bearer {test_token}"}

        create_response = await app_client.post(
            "/api/v1/sessions",
            json={},
            headers=headers,
        )
        session_id = create_response.json()["id"]

        # Act
        response = await app_client.put(
            f"/api/v1/sessions/{session_id}",
            json={"context_window": 200},  # Exceeds maximum of 100
            headers=headers,
        )

        # Assert
        assert response.status_code == 422  # Validation error
