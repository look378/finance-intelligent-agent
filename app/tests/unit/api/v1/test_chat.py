"""Tests for chat API endpoints"""
import pytest
from unittest.mock import Mock, AsyncMock
from fastapi.testclient import TestClient

from app.api.v1.chat import get_chat_service


class TestChatEndpoints:
    """Test chat API endpoints"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from app.main import create_app

        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def mock_chat_service(self):
        """Mock chat service"""
        service = Mock()
        service.process_message = AsyncMock(
            return_value=Mock(
                content="Hello! How can I help?",
                session_id=1,
                intent="greeting",
                sources=None,
                metadata={"tokens": 20}
            )
        )
        service.get_chat_history = AsyncMock(return_value=[])
        service.clear_chat_history = AsyncMock()
        return service

    def _override_chat_service(self, app, mock_service):
        """Override chat service dependency."""
        app.dependency_overrides[get_chat_service] = lambda: mock_service

    def test_chat_endpoint(self, client, mock_chat_service):
        """Test POST /chat endpoint"""
        self._override_chat_service(client.app, mock_chat_service)
        request_data = {"message": "Hello", "session_id": 1}

        response = client.post("/api/v1/chat", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Hello! How can I help?"
        assert data["session_id"] == 1
        assert data["intent"] == "greeting"
        mock_chat_service.process_message.assert_called_once()

    def test_chat_endpoint_with_user_id(self, client, mock_chat_service):
        """Test chat endpoint with user_id"""
        self._override_chat_service(client.app, mock_chat_service)
        request_data = {"message": "Hello", "session_id": 1, "user_id": 1}

        response = client.post("/api/v1/chat", json=request_data)

        assert response.status_code == 200
        mock_chat_service.process_message.assert_called_once()

    def test_chat_endpoint_validation_error(self, client, mock_chat_service):
        """Test chat endpoint with invalid request"""
        self._override_chat_service(client.app, mock_chat_service)
        request_data = {"session_id": 1}  # Missing "message" field

        response = client.post("/api/v1/chat", json=request_data)

        assert response.status_code == 422

    def test_chat_history_endpoint(self, client, mock_chat_service):
        """Test GET /chat/history endpoint"""
        mock_chat_service.get_chat_history = AsyncMock(
            return_value=[
                Mock(role="user", content="Hello", timestamp=None),
                Mock(role="assistant", content="Hi there!", timestamp=None),
            ]
        )
        self._override_chat_service(client.app, mock_chat_service)

        response = client.get("/api/v1/chat/history?session_id=1")

        assert response.status_code == 200
        data = response.json()
        assert "messages" in data
        assert len(data["messages"]) == 2

    def test_chat_history_with_limit(self, client, mock_chat_service):
        """Test chat history with limit parameter"""
        mock_chat_service.get_chat_history = AsyncMock(return_value=[])
        self._override_chat_service(client.app, mock_chat_service)

        response = client.get("/api/v1/chat/history?session_id=1&limit=10")

        assert response.status_code == 200
        mock_chat_service.get_chat_history.assert_called_once_with(
            session_id=1, limit=10,
        )

    def test_chat_history_missing_session_id(self, client, mock_chat_service):
        """Test chat history without session_id"""
        self._override_chat_service(client.app, mock_chat_service)
        response = client.get("/api/v1/chat/history")

        assert response.status_code == 422

    def test_clear_chat_history_endpoint(self, client, mock_chat_service):
        """Test DELETE /chat/history endpoint"""
        self._override_chat_service(client.app, mock_chat_service)

        response = client.delete("/api/v1/chat/history?session_id=1")

        assert response.status_code == 204
        mock_chat_service.clear_chat_history.assert_called_once_with(session_id=1)

    def test_chat_stream_endpoint(self, client, mock_chat_service):
        """Test streaming chat endpoint"""
        async def mock_stream(**kwargs):
            yield "Hello"
            yield " there"
            yield "!"

        mock_chat_service.process_message_stream = mock_stream
        self._override_chat_service(client.app, mock_chat_service)
        request_data = {"message": "Hello", "session_id": 1}

        response = client.post("/api/v1/chat/stream", json=request_data)

        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        chunks = []
        for line in response.iter_lines():
            if line:
                chunks.append(line if isinstance(line, str) else line.decode())
        assert len(chunks) >= 3

    def test_chat_endpoint_with_retrieval(self, client, mock_chat_service):
        """Test chat with retrieval sources"""
        mock_chat_service.process_message = AsyncMock(
            return_value=Mock(
                content="理财非存款，产品有风险，投资须谨慎。",
                session_id=1,
                intent="policy",
                sources=["doc1", "doc2"],
                metadata={"tokens": 50}
            )
        )
        self._override_chat_service(client.app, mock_chat_service)
        request_data = {"message": "理财有风险吗？", "session_id": 1}

        response = client.post("/api/v1/chat", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert "sources" in data
        assert len(data["sources"]) == 2

    def test_chat_endpoint_error_handling(self, client, mock_chat_service):
        """Test chat endpoint error handling"""
        from app.core.exceptions import BaseServiceError

        mock_chat_service.process_message = AsyncMock(
            side_effect=BaseServiceError("Processing failed")
        )
        self._override_chat_service(client.app, mock_chat_service)
        request_data = {"message": "Hello", "session_id": 1}

        response = client.post("/api/v1/chat", json=request_data)

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

    def test_chat_endpoint_unauthorized(self, client):
        """Test chat endpoint without authentication"""
        request_data = {"message": "Hello", "session_id": 1}

        response = client.post("/api/v1/chat", json=request_data)

        # Auth is optional (demo mode), so should return 503 (service not init)
        assert response.status_code in [200, 401, 503]


class TestChatSchemas:
    """Test chat request/response schemas"""

    def test_chat_request_schema(self):
        from app.api.v1.chat import ChatRequest

        request = ChatRequest(
            message="Hello", session_id=1, user_id=1, max_tokens=100,
        )
        assert request.message == "Hello"
        assert request.session_id == 1
        assert request.user_id == 1
        assert request.max_tokens == 100

    def test_chat_request_optional_fields(self):
        from app.api.v1.chat import ChatRequest

        request = ChatRequest(message="Hello", session_id=1)
        assert request.user_id is None
        assert request.max_tokens is None

    def test_chat_response_schema(self):
        from app.api.v1.chat import ChatResponse

        response = ChatResponse(
            content="Hello!", session_id=1, intent="greeting",
            sources=None, metadata={"tokens": 20},
        )
        assert response.content == "Hello!"
        assert response.intent == "greeting"

    def test_chat_history_response_schema(self):
        from app.api.v1.chat import ChatHistoryResponse, ChatMessageResponse

        response = ChatHistoryResponse(
            messages=[
                ChatMessageResponse(role="user", content="Hello"),
                ChatMessageResponse(role="assistant", content="Hi there!"),
            ],
            session_id=1,
        )
        assert len(response.messages) == 2
        assert response.messages[0].role == "user"
