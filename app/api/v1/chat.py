"""
聊天 API 端点。

提供消息处理、流式响应与历史记录管理等 REST API。
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.api.rate_limit import check_rate_limit
from app.models.database.user import User
from app.services.chat.chat_service import ChatService
from app.services.chat.factory import ChatServiceFactory
from app.services.llm import LLMFactory
from app.services.embeddings import EmbeddingFactory
from app.services.retrieval import RetrievalFactory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["聊天"])

# Global chat service instance (initialized on startup)
_chat_service: Optional[ChatService] = None


async def initialize_chat_service(db: AsyncSession):
    """使用全部依赖初始化 RAG 聊天服务。"""
    global _chat_service

    llm_service = LLMFactory.create_from_settings()

    from app.repositories.message_repository import MessageRepository
    from app.repositories.session_repository import SessionRepository

    message_repo = MessageRepository(db)
    session_repo = SessionRepository(db)

    # ── Retrieval pipeline ────────────────────────────────────────────────
    retrieval_pipeline = None
    try:
        from app.config.settings import get_settings as _get_settings

        _settings = _get_settings()
        embedding_service = EmbeddingFactory.create_from_settings()
        qdrant_client = RetrievalFactory.create_vector_client(
            client_type="qdrant",
            url=_settings.VECTOR_DB_URL,
            collection_name=_settings.VECTOR_COLLECTION_NAME,
            api_key=_settings.VECTOR_API_KEY,
            embedding_service=embedding_service,
        )
        retrieval_pipeline = {
            "hybrid_search": RetrievalFactory.create_hybrid_search(
                vector_client=qdrant_client,
            ),
        }
    except Exception as e:
        logger.warning("Failed to initialize retrieval pipeline: %s", e)

    # Initialize reranker
    try:
        from app.config.settings import get_settings

        settings = get_settings()
        if settings.RERANKER_ENABLED and retrieval_pipeline is not None:
            reranker = RetrievalFactory.create_reranker_from_settings(
                llm_service=llm_service,
            )
            retrieval_pipeline["reranker"] = reranker
    except Exception as e:
        logger.warning("Failed to initialize reranker: %s", e)

    # ── GraphRAG services ─────────────────────────────────────────────────
    graph_retrieval_service = None
    global_search_service = None
    multi_path_fusion = None

    try:
        from app.config.settings import get_settings

        settings = get_settings()
        if settings.GRAPH_RAG_ENABLED:
            from app.services.graph import GraphFactory
            from app.services.graph.retrieval import (
                GraphRetrievalService,
                MultiPathRetrievalFusion,
            )

            graph_client = GraphFactory.create_from_settings()
            if graph_client:
                await graph_client.connect()

                graph_retrieval_service = GraphRetrievalService(
                    text_to_cypher=None,
                    graph_embedding_search=None,
                )
                global_search_service = None
                multi_path_fusion = MultiPathRetrievalFusion(
                    vector_weight=1.0 - settings.GRAPH_RAG_FUSION_WEIGHT,
                    graph_weight=settings.GRAPH_RAG_FUSION_WEIGHT,
                )

                if settings.GRAPH_RAG_TEXT_TO_CYPHER_ENABLED:
                    from app.services.graph.retrieval import TextToCypherService

                    graph_retrieval_service._cypher = TextToCypherService(
                        llm_service=llm_service,
                        graph_client=graph_client,
                    )

                from app.services.graph.retrieval import GraphEmbeddingSearch

                graph_embedding = GraphEmbeddingSearch(
                    graph_client=graph_client,
                    embedding_service=embedding_service,
                    max_hops=settings.GRAPH_RAG_MAX_HOPS,
                )
                graph_retrieval_service._embedding_search = graph_embedding

                if settings.GRAPH_RAG_COMMUNITY_ENABLED:
                    from app.services.graph.community import GlobalSearchService

                    global_search_service = GlobalSearchService(
                        graph_client=graph_client,
                        embedding_service=embedding_service,
                    )
    except Exception as e:
        logger.warning("Failed to initialize GraphRAG services: %s", e)

    # ── Guardrails ────────────────────────────────────────────────────────
    guardrail_service = None
    try:
        from app.services.guardrails.factory import GuardrailFactory

        guardrail_service = GuardrailFactory.create_from_settings()
    except Exception as e:
        logger.warning("Failed to initialize guardrails: %s", e)

    # ── Build ChatService via factory (graph constructed internally) ──────
    _chat_service = ChatServiceFactory.create_with_defaults(
        llm_service=llm_service,
        message_repo=message_repo,
        session_repo=session_repo,
        memory_type="optimized",
        intent_type="hybrid",
        retrieval_pipeline=retrieval_pipeline,
        graph_retrieval_service=graph_retrieval_service,
        global_search_service=global_search_service,
        multi_path_fusion=multi_path_fusion,
        guardrail_service=guardrail_service,
    )


def get_chat_service() -> ChatService:
    """获取聊天服务实例。未初始化时抛出 503。"""
    global _chat_service
    if _chat_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service not initialized. Please ensure all dependencies are configured.",
        )
    return _chat_service


# ── Request / Response schemas ────────────────────────────────────────────


class ChatRequest(BaseModel):
    """聊天请求体。"""

    message: str = Field(..., min_length=1, description="用户消息")
    session_id: int = Field(..., gt=0, description="会话 ID")
    user_id: Optional[int] = Field(None, gt=0, description="用户 ID（可选）")
    max_tokens: Optional[int] = Field(None, gt=0, le=4096, description="回复最大 token 数")


class ChatResponse(BaseModel):
    """聊天响应体。"""

    content: str
    session_id: int
    intent: str
    sources: Optional[List[str]] = None
    metadata: Optional[dict] = None
    dialogue_state: Optional[dict] = None


class ChatMessageResponse(BaseModel):
    """历史记录中的聊天消息。"""

    role: str
    content: str
    timestamp: Optional[str] = None


class ChatHistoryResponse(BaseModel):
    """聊天历史响应。"""

    messages: List[ChatMessageResponse]
    session_id: int


# ── Endpoints ─────────────────────────────────────────────────────────────


@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK, summary="发送聊天消息")
async def chat(
    request: ChatRequest,
    http_req: Request,
    current_user: Optional[User] = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
):
    """处理聊天消息并生成回复。"""
    check_rate_limit(http_req)

    try:
        user_id = current_user.id if current_user else request.user_id

        response = await chat_service.process_message(
            session_id=request.session_id,
            message=request.message,
            user_id=user_id or 0,
            max_tokens=request.max_tokens,
        )

        # 从 metadata 构建对话状态
        dialogue_state = None
        if response.metadata:
            dialogue_state = {
                "phase": response.intent,
                "pending_slots": response.metadata.get("pending_slots", []),
                "filled_slots": response.metadata.get("filled_slots", {}),
            }

        return ChatResponse(
            content=response.content,
            session_id=response.session_id,
            intent=response.intent,
            sources=response.sources,
            metadata=response.metadata,
            dialogue_state=dialogue_state,
        )

    except Exception as e:
        logger.error("Failed to process chat message: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="消息处理失败",
        )


@router.post("/stream", summary="流式发送聊天消息")
async def chat_stream(
    request: ChatRequest,
    http_req: Request,
    current_user: Optional[User] = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
):
    """以流式方式处理聊天消息并生成回复。"""
    check_rate_limit(http_req)

    try:
        user_id = current_user.id if current_user else request.user_id

        async def generate():
            async for chunk in chat_service.process_message_stream(
                session_id=request.session_id,
                message=request.message,
                user_id=user_id or 0,
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )

    except Exception as e:
        logger.error("Failed to process streaming message: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="流式消息处理失败",
        )


@router.get("/history", response_model=ChatHistoryResponse, summary="获取聊天历史")
async def get_chat_history(
    session_id: int = Query(..., gt=0, description="会话 ID"),
    limit: int = Query(50, gt=0, le=100, description="返回的最大消息数"),
    current_user: Optional[User] = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
):
    """获取指定会话的聊天历史。"""
    try:
        messages = await chat_service.get_chat_history(
            session_id=session_id,
            limit=limit,
        )

        return ChatHistoryResponse(
            messages=[
                ChatMessageResponse(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp,
                )
                for msg in messages
            ],
            session_id=session_id,
        )

    except Exception as e:
        logger.error("Failed to get chat history: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="获取历史记录失败",
        )


@router.delete("/history", status_code=status.HTTP_204_NO_CONTENT, summary="清空聊天历史")
async def clear_chat_history(
    session_id: int = Query(..., gt=0, description="会话 ID"),
    current_user: Optional[User] = Depends(get_current_user),
    chat_service: ChatService = Depends(get_chat_service),
):
    """清空指定会话的聊天历史。"""
    try:
        await chat_service.clear_chat_history(session_id=session_id)
    except Exception as e:
        logger.error("Failed to clear chat history: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="清空历史记录失败",
        )
