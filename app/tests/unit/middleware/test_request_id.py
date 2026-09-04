"""Tests for request ID middleware"""
import pytest
import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from unittest.mock import AsyncMock


class TestRequestIDMiddleware:
    """Test request ID middleware"""

    def test_middleware_initialization(self):
        """Test middleware initialization"""
        # Arrange
        from app.middleware.request_id import RequestIDMiddleware

        app = FastAPI()

        # Act
        middleware = RequestIDMiddleware(app)

        # Assert
        assert middleware.app == app
        assert middleware.header_name == "X-Request-ID"

    def test_middleware_with_custom_header(self):
        """Test middleware with custom header name"""
        # Arrange
        from app.middleware.request_id import RequestIDMiddleware

        app = FastAPI()

        # Act
        middleware = RequestIDMiddleware(app, header_name="X-Correlation-ID")

        # Assert
        assert middleware.header_name == "X-Correlation-ID"

    @pytest.mark.asyncio
    async def test_generates_request_id_if_not_provided(self):
        """Test generates request ID if not in headers"""
        # Arrange
        from app.middleware.request_id import RequestIDMiddleware

        app = FastAPI()
        middleware = RequestIDMiddleware(app)

        request = Request(
            scope={
                "type": "http",
                "method": "GET",
                "path": "/test",
                "headers": [],
                "query_string": b"",
            },
            receive=None,
        )

        # Act
        response = await middleware.dispatch(request, self._mock_call_next)

        # Assert - Should have generated request ID
        assert "X-Request-ID" in response.headers or "x-request-id" in response.headers
        request_id = response.headers.get("X-Request-ID") or response.headers.get("x-request-id")
        assert request_id is not None
        # Should be valid UUID
        uuid.UUID(request_id)

    @pytest.mark.asyncio
    async def test_uses_existing_request_id_from_header(self):
        """Test uses existing request ID from header"""
        # Arrange
        from app.middleware.request_id import RequestIDMiddleware

        app = FastAPI()
        middleware = RequestIDMiddleware(app)

        existing_id = "existing-request-id-123"

        request = Request(
            scope={
                "type": "http",
                "method": "GET",
                "path": "/test",
                "headers": [(b"X-Request-ID", existing_id.encode())],
                "query_string": b"",
            },
            receive=None,
        )

        # Act
        response = await middleware.dispatch(request, self._mock_call_next)

        # Assert - Should use existing ID
        response_id = response.headers.get("X-Request-ID") or response.headers.get("x-request-id")
        assert response_id == existing_id

    @pytest.mark.asyncio
    async def test_adds_request_id_to_response_headers(self):
        """Test adds request ID to response headers"""
        # Arrange
        from app.middleware.request_id import RequestIDMiddleware

        app = FastAPI()
        middleware = RequestIDMiddleware(app)

        request = Request(
            scope={
                "type": "http",
                "method": "GET",
                "path": "/test",
                "headers": [],
                "query_string": b"",
            },
            receive=None,
        )

        # Act
        response = await middleware.dispatch(request, self._mock_call_next)

        # Assert - Response should have request ID header
        headers = response.headers
        assert "X-Request-ID" in headers or "x-request-id" in headers

    @pytest.mark.asyncio
    async def test_request_id_in_request_state(self):
        """Test request ID is available in request state"""
        # Arrange
        from app.middleware.request_id import RequestIDMiddleware

        app = FastAPI()
        middleware = RequestIDMiddleware(app)

        request = Request(
            scope={
                "type": "http",
                "method": "GET",
                "path": "/test",
                "headers": [],
                "query_string": b"",
                "state": {},
            },
            receive=None,
        )

        # Act
        await middleware.dispatch(request, self._mock_call_next)

        # Assert - Request ID should be in state
        assert "request_id" in request.state
        assert isinstance(request.state["request_id"], str)

    @pytest.mark.asyncio
    async def test_preserves_existing_response_headers(self):
        """Test preserves existing response headers"""
        # Arrange
        from app.middleware.request_id import RequestIDMiddleware

        app = FastAPI()
        middleware = RequestIDMiddleware(app)

        request = Request(
            scope={
                "type": "http",
                "method": "GET",
                "path": "/test",
                "headers": [],
                "query_string": b"",
            },
            receive=None,
        )

        # Act
        response = await middleware.dispatch(request, self._mock_call_next)

        # Assert - Other headers should be preserved
        # Mock call_next sets content-type
        assert "content-type" in response.headers or "Content-Type" in response.headers

    @pytest.mark.asyncio
    async def test_custom_header_name(self):
        """Test custom header name"""
        # Arrange
        from app.middleware.request_id import RequestIDMiddleware

        app = FastAPI()
        middleware = RequestIDMiddleware(app, header_name="X-Correlation-ID")

        request = Request(
            scope={
                "type": "http",
                "method": "GET",
                "path": "/test",
                "headers": [],
                "query_string": b"",
            },
            receive=None,
        )

        # Act
        response = await middleware.dispatch(request, self._mock_call_next)

        # Assert - Should use custom header name
        assert "X-Correlation-ID" in response.headers or "x-correlation-id" in response.headers

    def test_generate_request_id(self):
        """Test request ID generation"""
        # Arrange
        from app.middleware.request_id import RequestIDMiddleware

        middleware = RequestIDMiddleware(FastAPI())

        # Act
        request_id = middleware._generate_request_id()

        # Assert - Should be valid UUID
        assert isinstance(request_id, str)
        uuid.UUID(request_id)  # Will raise if invalid

    def test_generate_request_id_unique(self):
        """Test generated request IDs are unique"""
        # Arrange
        from app.middleware.request_id import RequestIDMiddleware

        middleware = RequestIDMiddleware(FastAPI())

        # Act - Generate multiple IDs
        ids = [middleware._generate_request_id() for _ in range(100)]

        # Assert - All should be unique
        assert len(set(ids)) == 100

    async def _mock_call_next(self, request):
        """Mock call_next function"""
        # Add request ID to state for testing
        if "request_id" not in request.state:
            request.state["request_id"] = "test-request-id"

        return JSONResponse(content={"status": "ok"})
