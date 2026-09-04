"""Tests for Prometheus metrics middleware"""
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from unittest.mock import AsyncMock, Mock


class TestPrometheusMetrics:
    """Test Prometheus metrics collection"""

    def test_metrics_initialization(self):
        """Test metrics middleware initialization"""
        # Arrange
        from app.middleware.metrics import PrometheusMiddleware

        app = FastAPI()

        # Act
        middleware = PrometheusMiddleware(app)

        # Assert
        assert middleware.app == app
        assert middleware.app_name == "finance_agent"

    def test_metrics_with_custom_app_name(self):
        """Test metrics with custom app name"""
        # Arrange
        from app.middleware.metrics import PrometheusMiddleware

        app = FastAPI()

        # Act
        middleware = PrometheusMiddleware(app, app_name="custom_app")

        # Assert
        assert middleware.app_name == "custom_app"

    @pytest.mark.asyncio
    async def test_tracks_request_count(self):
        """Test tracking request count"""
        # Arrange
        from app.middleware.metrics import PrometheusMiddleware

        app = FastAPI()
        middleware = PrometheusMiddleware(app)

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

        # Act - Make multiple requests
        await middleware.dispatch(request, self._mock_call_next)
        await middleware.dispatch(request, self._mock_call_next)

        # Assert - Should track count
        assert middleware.request_count.get(
            {"method": "GET", "endpoint": "/test", "status": "200"}
        )._value._value == 2

    @pytest.mark.asyncio
    async def test_tracks_request_latency(self):
        """Test tracking request latency"""
        # Arrange
        from app.middleware.metrics import PrometheusMiddleware

        app = FastAPI()
        middleware = PrometheusMiddleware(app)

        request = Request(
            scope={
                "type": "http",
                "method": "POST",
                "path": "/api/v1/chat",
                "headers": [],
                "query_string": b"",
            },
            receive=None,
        )

        # Act
        await middleware.dispatch(request, self._mock_call_next)

        # Assert - Should track latency histogram
        samples = list(middleware.request_latency.collect())[0].samples
        assert len(samples) > 0

    @pytest.mark.asyncio
    async def test_tracks_active_requests(self):
        """Test tracking active requests (gauge)"""
        # Arrange
        from app.middleware.metrics import PrometheusMiddleware

        app = FastAPI()
        middleware = PrometheusMiddleware(app)

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
        import asyncio
        task = asyncio.create_task(middleware.dispatch(request, self._mock_call_next))
        await asyncio.sleep(0.01)  # Let it start
        await task

        # Assert - Should track gauge (0 when complete)
        assert middleware.active_requests._value._value == 0

    @pytest.mark.asyncio
    async def test_tracks_errors_by_status(self):
        """Test tracking errors by status code"""
        # Arrange
        from app.middleware.metrics import PrometheusMiddleware

        app = FastAPI()
        middleware = PrometheusMiddleware(app)

        async def failing_call_next(request):
            return JSONResponse(content={"error": "Not found"}, status_code=404)

        request = Request(
            scope={
                "type": "http",
                "method": "GET",
                "path": "/notfound",
                "headers": [],
                "query_string": b"",
            },
            receive=None,
        )

        # Act
        await middleware.dispatch(request, failing_call_next)

        # Assert - Should track 404 errors
        assert middleware.request_count.get(
            {"method": "GET", "endpoint": "/notfound", "status": "404"}
        )._value._value == 1

    def test_metrics_endpoint(self):
        """Test /metrics endpoint"""
        # Arrange
        from app.middleware.metrics import PrometheusMiddleware, metrics_endpoint

        app = FastAPI()
        middleware = PrometheusMiddleware(app)

        # Act - Get metrics
        from starlette.testclient import TestClient
        client = TestClient(app)
        response = client.get("/metrics")

        # Assert
        assert response.status_code == 200
        # Should have Prometheus format
        assert "http_requests_total" in response.text or "request_count" in response.text


class TestMetricsFormats:
    """Test metrics output formats"""

    def test_prometheus_format(self):
        """Test Prometheus text format"""
        # Arrange
        from prometheus_client import Counter

        counter = Counter("test_requests_total", "Test requests", ["method"])

        # Act
        counter.labels(method="GET").inc()

        # Assert
        from prometheus_client import exposition
        output = exposition.generate_latest(counter).decode()
        assert "test_requests_total" in output

    def test_histogram_buckets(self):
        """Test histogram has proper buckets"""
        # Arrange
        from prometheus_client import Histogram

        histogram = Histogram(
            "request_latency_seconds",
            "Request latency",
            ["endpoint"]
        )

        # Act - Observe some latencies
        histogram.labels(endpoint="/test").observe(0.1)
        histogram.labels(endpoint="/test").observe(0.5)
        histogram.labels(endpoint="/test").observe(1.5)

        # Assert
        samples = list(histogram.collect())[0].samples
        # Should have bucket samples
        bucket_samples = [s for s in samples if s.name.endswith("_bucket")]
        assert len(bucket_samples) > 0


class TestMetricsLabels:
    """Test metrics labeling"""

    def test_default_labels(self):
        """Test default metric labels"""
        # Arrange
        from app.middleware.metrics import PrometheusMiddleware

        app = FastAPI()
        middleware = PrometheusMiddleware(app)

        # Assert - Should have standard labels
        assert hasattr(middleware, "request_count")
        assert hasattr(middleware, "request_latency")
        assert hasattr(middleware, "active_requests")

    def test_custom_labels_allowed(self):
        """Test custom labels can be added"""
        # Arrange & Act
        from prometheus_client import Counter

        counter = Counter(
            "custom_metric",
            "Custom metric",
            ["label1", "label2"]
        )

        # Assert
        counter.labels(label1="value1", label2="value2").inc()
        assert counter.labels(label1="value1", label2="value2")._value._value == 1

    async def _mock_call_next(self, request):
        """Mock call_next function"""
        return JSONResponse(content={"status": "ok"})


class TestMetricsIntegration:
    """Test metrics integration"""

    @pytest.mark.asyncio
    async def test_full_request_tracked(self):
        """Test complete request is tracked"""
        # Arrange
        from app.middleware.metrics import PrometheusMiddleware

        app = FastAPI()
        middleware = PrometheusMiddleware(app)

        @app.get("/test")
        async def test_endpoint():
            return {"message": "test"}

        request = Request(
            scope={
                "type": "http",
                "method": "GET",
                "path": "/test",
                "headers": [],
                "query_string": b"",
                "app": app,
            },
            receive=None,
        )

        # Act
        await middleware.dispatch(request, self._mock_call_next)

        # Assert - Metrics should be recorded
        assert middleware.request_count._value._value > 0
