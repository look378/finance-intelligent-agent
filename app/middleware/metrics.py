"""
Prometheus metrics middleware.

Collects and exposes metrics for monitoring.
"""
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from prometheus_client.openmetrics.exposition import generate_latest as generate_latest_openmetrics
from typing import Dict
import time
import logging


logger = logging.getLogger(__name__)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Middleware to collect Prometheus metrics.

    Tracks:
    - Request count (by method, endpoint, status)
    - Request latency (histogram)
    - Active requests (gauge)
    """

    def __init__(self, app, app_name: str = "finance_agent"):
        super().__init__(app)
        self.app_name = app_name

        # Define metrics
        self.request_count = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
            registry=REGISTRY,
        )

        self.request_latency = Histogram(
            "http_request_duration_seconds",
            "HTTP request latency",
            ["method", "endpoint"],
            registry=REGISTRY,
        )

        self.active_requests = Gauge(
            "http_requests_active",
            "Active HTTP requests",
            registry=REGISTRY,
        )

    async def dispatch(self, request: Request, call_next):
        """
        Process request and collect metrics.

        Args:
            request: Incoming request
            call_next: Next middleware/route handler

        Returns:
            Response: HTTP response
        """
        # Get endpoint path
        endpoint = request.url.path

        # Increment active requests
        self.active_requests.inc()

        # Start timer
        start_time = time.time()

        try:
            # Process request
            response: Response = await call_next(request)

            # Record metrics
            status = str(response.status_code)
            self.request_count.labels(
                method=request.method,
                endpoint=endpoint,
                status=status,
            ).inc()

            # Record latency
            latency = time.time() - start_time
            self.request_latency.labels(
                method=request.method,
                endpoint=endpoint,
            ).observe(latency)

            return response

        finally:
            # Decrement active requests
            self.active_requests.dec()


def metrics_endpoint(request: Request):
    """
    FastAPI endpoint to expose Prometheus metrics.

    Returns:
        Response: Prometheus metrics text format
    """
    from fastapi.responses import Response

    # Generate metrics
    metrics = generate_latest(REGISTRY)

    return Response(
        content=metrics,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
