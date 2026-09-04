"""
FastAPI 应用入口。

组装 FastAPI 应用、配置中间件并挂载全部路由。
"""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html

from app.config.settings import settings
from app.config.logging import logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    应用生命周期管理器。

    处理启动与关闭事件。
    """
    # 启动
    logger.info(
        "启动金融智能客服",
        environment=settings.ENVIRONMENT,
        debug=settings.DEBUG,
    )

    # 初始化聊天服务
    try:
        from app.api.database import async_session_maker
        from app.api.v1.chat import initialize_chat_service
        async with async_session_maker() as db:
            await initialize_chat_service(db)
        logger.info("聊天服务初始化成功")
    except Exception as e:
        logger.warning("聊天服务初始化失败: %s", e)

    yield
    # 关闭
    logger.info("金融智能客服已关闭")


def create_app() -> FastAPI:
    """
    创建并配置 FastAPI 应用。

    返回:
        FastAPI: 配置完成的 FastAPI 应用实例
    """
    app = FastAPI(
        title="金融智能客服 API",
        description=(
            "基于 LangGraph 的金融财富管理智能客服。\n\n"
            "支持理财咨询、基金查询、保险咨询、持仓查询、风险测评与收益测算，"
            "内置投资者适当性管理、金融合规护栏（承诺收益拦截、风险提示注入）与结构化审计日志。"
        ),
        version="1.0.0",
        # 禁用 FastAPI 自动生成的英文文档路由，改用下方自定义的中文版本
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )

    # 配置 CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 请求 ID 中间件
    from app.middleware.request_id import RequestIDMiddleware
    app.add_middleware(RequestIDMiddleware)

    # Prometheus 指标中间件
    if settings.ENABLE_METRICS:
        from app.middleware.metrics import PrometheusMiddleware
        app.add_middleware(PrometheusMiddleware)

    # OpenTelemetry 链路追踪
    if settings.ENABLE_TRACING:
        from app.middleware.tracing import setup_tracing
        setup_tracing(app=app, endpoint=settings.OTEL_ENDPOINT)

    # 挂载 API v1 路由
    from app.api.v1.router import api_router
    app.include_router(api_router, prefix=settings.API_PREFIX)

    # 健康检查端点
    @app.get("/health", tags=["系统"])
    async def health_check():
        """基础健康检查端点"""
        return {
            "status": "healthy",
            "environment": settings.ENVIRONMENT,
        }

    # 中文 Swagger UI 文档页（覆盖默认英文 /docs）
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html() -> HTMLResponse:
        """中文版交互式 API 文档"""
        response = get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title="金融智能客服 API 文档",
            swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
            swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
            swagger_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
        )
        # get_swagger_ui_html 可能返回 HTMLResponse 或 str，统一转成字符串
        if isinstance(response, HTMLResponse):
            html = response.body.decode("utf-8")
        else:
            html = str(response)
        # 注入中文 locale（Swagger UI 5.x 按钮级翻译）
        marker = '"layout": "BaseLayout",'
        if marker in html:
            html = html.replace(marker, marker + "\n    locale: 'zh-cn',")
        # 注入中文翻译补丁：用 MutationObserver 监听 DOM 变化，
        # 将 Swagger UI 的英文标题/按钮文本替换为中文（不改动元素结构）
        zh_patch = """
    <script>
    (function () {
        var translations = {
            'Schemas': '数据模型',
            'Authorize': '授权',
            'Execute': '执行',
            'Try it out': '试一试',
            'Cancel': '取消',
            'Clear': '清除',
            'Parameters': '参数',
            'Request body': '请求体',
            'Responses': '响应',
            'Code': '状态码',
            'Description': '说明',
            'Value': '值',
            'Parameter name': '参数名',
            'No examples': '无示例',
            'Download': '下载',
            'available': '可用',
            'Example Value': '示例值',
            'Schema': '模型',
            'Model': '模型',
            'default': '默认分组',
            'General': '通用',
        };
        function replaceText(node) {
            if (node.nodeType === Node.TEXT_NODE) {
                var text = node.nodeValue.trim();
                if (text && translations[text] !== undefined && text !== translations[text]) {
                    node.nodeValue = node.nodeValue.replace(text, translations[text]);
                }
                return;
            }
            // 只处理标题/按钮等叶子文本，避免破坏链接与代码块
            var tag = node.nodeName ? node.nodeName.toLowerCase() : '';
            if (['h1', 'h2', 'h3', 'h4', 'th', 'button', 'span', 'a', 'td'].indexOf(tag) >= 0) {
                var directText = '';
                node.childNodes.forEach(function (c) {
                    if (c.nodeType === Node.TEXT_NODE) directText += c.nodeValue;
                });
                var trimmed = directText.trim();
                if (trimmed && translations[trimmed] !== undefined && trimmed !== translations[trimmed]) {
                    node.childNodes.forEach(function (c) {
                        if (c.nodeType === Node.TEXT_NODE) {
                            c.nodeValue = c.nodeValue.replace(trimmed, translations[trimmed]);
                        }
                    });
                }
            }
        }
        function scan(root) {
            var walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT, null);
            var nodes = [];
            while (walker.nextNode()) nodes.push(walker.currentNode);
            nodes.forEach(replaceText);
        }
        var target = document.getElementById('swagger-ui') || document.body;
        scan(target);
        var observer = new MutationObserver(function (mutations) {
            mutations.forEach(function (m) {
                m.addedNodes.forEach(function (n) { if (n.nodeType === 1) scan(n); });
            });
        });
        observer.observe(target, { childList: true, subtree: true });
        // Swagger UI 异步渲染，多次重扫兜底
        [1000, 3000, 6000].forEach(function (ms) {
            setTimeout(function () { scan(document.getElementById('swagger-ui') || document.body); }, ms);
        });
    })();
    </script>
"""
        html = html.replace("</body>", zh_patch + "</body>")
        return HTMLResponse(content=html)

    @app.get("/redoc", include_in_schema=False)
    async def custom_redoc_html() -> HTMLResponse:
        """中文版 ReDoc 文档页"""
        from fastapi.openapi.docs import get_redoc_html

        return get_redoc_html(
            openapi_url=app.openapi_url,
            title="金融智能客服 API 文档",
            redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
            redoc_favicon_url="https://fastapi.tiangolo.com/img/favicon.png",
            with_google_fonts=False,
        )

    # Prometheus 指标端点
    if settings.ENABLE_METRICS:
        from app.middleware.metrics import metrics_endpoint
        app.add_route("/metrics", metrics_endpoint)

    # 根端点：返回中文聊天使用页面
    @app.get("/", include_in_schema=False, tags=["系统"])
    async def root():
        """金融智能客服聊天使用页面"""
        from fastapi.responses import FileResponse

        return FileResponse("app/static/chat.html")

    # API 基础信息端点（供程序化访问）
    @app.get("/api/info", tags=["系统"])
    async def api_info():
        """返回 API 基本信息"""
        return {
            "name": "金融智能客服 API",
            "version": "1.0.0",
            "status": "运行中",
            "docs": "/docs",
            "chat_ui": "/",
        }

    # 全局异常处理器
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """未捕获异常的全局处理器"""
        logger.error(
            "未捕获异常",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "服务器内部错误",
                "error": "SERVER_ERROR" if not settings.DEBUG else str(exc),
            },
        )

    return app


# 创建应用实例
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
    )
