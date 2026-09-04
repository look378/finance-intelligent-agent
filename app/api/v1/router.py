"""
API router aggregation for v1 endpoints.

Combines all v1 routers into a single router for inclusion in the main app.
"""
from fastapi import APIRouter

from app.api.v1 import auth, sessions, documents, chat, feedback

api_router = APIRouter()

api_router.include_router(auth.router)
api_router.include_router(sessions.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)
api_router.include_router(feedback.router)

# Conditionally include graph router when GraphRAG is enabled
try:
    from app.config.settings import get_settings
    settings = get_settings()
    if settings.GRAPH_RAG_ENABLED:
        from app.api.v1 import graph
        api_router.include_router(graph.router)
except Exception:
    pass
