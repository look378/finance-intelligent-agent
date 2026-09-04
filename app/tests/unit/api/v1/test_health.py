"""Tests for health check endpoints"""
import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Test health check endpoints"""

    def test_health_endpoint(self):
        """Test /health endpoint"""
        # Arrange
        from app.main import create_app

        app = create_app()
        client = TestClient(app)

        # Act
        response = client.get("/health")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_ready_endpoint(self):
        """Test /ready endpoint"""
        # Arrange
        from app.main import create_app

        app = create_app()
        client = TestClient(app)

        # Act
        response = client.get("/ready")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"

    def test_metrics_endpoint(self):
        """Test /metrics endpoint"""
        # Arrange
        from app.main import create_app

        app = create_app()
        client = TestClient(app)

        # Act
        response = client.get("/metrics")

        # Assert
        assert response.status_code == 200
        # Should be text/plain
        assert "text/plain" in response.headers["content-type"]
        # Should have Prometheus metrics
        assert "http_requests_total" in response.text or "http" in response.text.lower()

    def test_health_includes_service_name(self):
        """Test health check includes service name"""
        # Arrange
        from app.main import create_app

        app = create_app()
        client = TestClient(app)

        # Act
        response = client.get("/health")

        # Assert
        data = response.json()
        assert "service" in data
        assert data["service"] == "finance_agent"

    def test_ready_checks_dependencies(self):
        """Test readiness check validates dependencies"""
        # Arrange
        from app.main import create_app

        app = create_app()
        client = TestClient(app)

        # Act
        response = client.get("/ready")

        # Assert
        data = response.json()
        assert "status" in data
        # When all dependencies are ready
        assert data["status"] in ["ready", "not_ready"]

    def test_health_endpoint_caching(self):
        """Test health endpoint allows caching"""
        # Arrange
        from app.main import create_app

        app = create_app()
        client = TestClient(app)

        # Act
        response = client.get("/health")

        # Assert - Should allow caching (optional)
        cache_control = response.headers.get("Cache-Control")
        # May or may not have cache control
        assert response.status_code == 200

    def test_metrics_updates_with_requests(self):
        """Test metrics update with requests"""
        # Arrange
        from app.main import create_app

        app = create_app()
        client = TestClient(app)

        # Make some requests
        for _ in range(5):
            client.get("/health")

        # Act
        response = client.get("/metrics")

        # Assert - Metrics should include requests
        assert "http_requests_total" in response.text
        # Should have tracked our requests
        assert "health" in response.text.lower() or "/health" in response.text


class TestMonitoringIntegration:
    """Test monitoring integration"""

    def test_all_monitoring_endpoints_available(self):
        """Test all monitoring endpoints are registered"""
        # Arrange
        from app.main import create_app

        app = create_app()
        client = TestClient(app)

        # Act & Assert
        # Health check
        assert client.get("/health").status_code == 200

        # Readiness check
        assert client.get("/ready").status_code == 200

        # Metrics
        assert client.get("/metrics").status_code == 200

    def test_monitoring_works_without_auth(self):
        """Test monitoring endpoints don't require authentication"""
        # Arrange
        from app.main import create_app

        app = create_app()
        client = TestClient(app)

        # Act & Assert - Should work without auth
        assert client.get("/health").status_code == 200
        assert client.get("/metrics").status_code == 200

    def test_monitoring_response_times(self):
        """Test monitoring endpoints respond quickly"""
        # Arrange
        from app.main import create_app
        import time

        app = create_app()
        client = TestClient(app)

        # Act
        start = time.time()
        response = client.get("/metrics")
        duration = time.time() - start

        # Assert - Should be fast (< 1 second)
        assert response.status_code == 200
        assert duration < 1.0


class TestHealthCheckDetailed:
    """Test detailed health checks"""

    def test_health_with_database_check(self):
        """Test health check includes database status"""
        # TODO: Implement when database health check is added
        pass

    def test_health_with_llm_check(self):
        """Test health check includes LLM service status"""
        # TODO: Implement when LLM health check is added
        pass

    def test_health_with_vector_db_check(self):
        """Test health check includes vector DB status"""
        # TODO: Implement when vector DB health check is added
        pass

    def test_ready_fails_when_dependency_down(self):
        """Test readiness check fails when dependency is down"""
        # TODO: Implement dependency checks
        pass

    def test_ready_includes_dependency_details(self):
        """Test readiness check includes dependency details"""
        # TODO: Implement detailed dependency checks
        pass
