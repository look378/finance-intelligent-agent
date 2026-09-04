"""
Simple rate limiter dependency using in-memory token bucket.

Provides rate limiting for API endpoints without external dependencies.
"""
from fastapi import Request, HTTPException, status
from typing import Dict
import time


class InMemoryRateLimiter:
    """
    In-memory rate limiter using token bucket algorithm.

    Simple implementation for demo purposes.
    """

    def __init__(self, requests_per_minute: int = 60):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute per IP
        """
        self.requests_per_minute = requests_per_minute
        self.bucket_size = requests_per_minute
        self.buckets: Dict[str, Dict] = {}
        self._bucket_ttl = 300.0  # 5 分钟无访问则清理，防止内存无限增长

    def _get_client_ip(self, request: Request) -> str:
        """Get client IP from the socket.

        不信任客户端可控的 X-Forwarded-For（可被伪造绕过限流）。
        如需支持代理后的真实 IP，应由可信反向代理统一改写后在此读取。
        """
        return request.client.host if request.client else "unknown"

    def _cleanup_stale_buckets(self, now: float) -> None:
        """清理超过 TTL 未访问的桶，防止内存持续增长。"""
        stale = [
            ip for ip, b in self.buckets.items()
            if now - b["last_update"] > self._bucket_ttl
        ]
        for ip in stale:
            del self.buckets[ip]

    def check_rate_limit(self, request: Request) -> None:
        """
        Check if request is within rate limit.

        Args:
            request: FastAPI request

        Raises:
            HTTPException: If rate limit exceeded
        """
        client_ip = self._get_client_ip(request)
        now = time.time()

        # 定期清理过期桶
        if len(self.buckets) > 1000:
            self._cleanup_stale_buckets(now)

        # Get or create bucket
        if client_ip not in self.buckets:
            self.buckets[client_ip] = {
                "tokens": self.bucket_size - 1,
                "last_update": now,
            }
            return

        bucket = self.buckets[client_ip]

        # Refill tokens based on elapsed time
        elapsed = now - bucket["last_update"]
        tokens_to_add = elapsed * (self.requests_per_minute / 60.0)
        bucket["tokens"] = min(self.bucket_size, bucket["tokens"] + tokens_to_add)
        bucket["last_update"] = now

        # Check if we have tokens
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
        else:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"请求过于频繁，请稍后再试。上限 {self.requests_per_minute} 次/分钟。",
            )


def create_rate_limiter() -> InMemoryRateLimiter:
    """Create rate limiter from settings."""
    from app.config.settings import get_settings

    settings = get_settings()
    return InMemoryRateLimiter(requests_per_minute=settings.RATE_LIMIT_PER_MINUTE)


# Global rate limiter instance
_rate_limiter = create_rate_limiter()


def check_rate_limit(request: Request):
    """
    FastAPI dependency for rate limiting.

    Args:
        request: FastAPI request

    Raises:
        HTTPException: If rate limit exceeded
    """
    _rate_limiter.check_rate_limit(request)
