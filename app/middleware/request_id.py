"""
请求 ID 中间件。

为每个请求生成唯一请求 ID，用于链路追踪与日志关联。
"""
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import uuid
import logging


logger = logging.getLogger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """
    为请求添加唯一 ID 的中间件。
    """

    def __init__(self, app, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self.header_name = header_name

    async def dispatch(self, request: Request, call_next):
        """
        处理请求并添加请求 ID。

        Args:
            request: 进入的请求
            call_next: 下一个中间件/路由处理器

        Returns:
            Response: 带请求 ID 的 HTTP 响应
        """
        # 忽略客户端传入的 X-Request-ID（避免日志注入/污染），一律服务端生成
        request_id = self._generate_request_id()

        # 存入请求状态供日志使用
        request.state.request_id = request_id

        # 处理请求
        response: Response = await call_next(request)

        # 写入响应头
        response.headers[self.header_name] = request_id

        return response

    def _generate_request_id(self) -> str:
        """
        生成唯一请求 ID。

        Returns:
            str: UUID v4
        """
        return str(uuid.uuid4())
