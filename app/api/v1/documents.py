"""
文档管理 API 端点。

提供文档上传、搜索与管理的 REST API。
"""
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Optional, List
from pydantic import BaseModel, Field
from io import BytesIO

from app.services.documents.ingestion import DocumentIngestionService
from app.services.embeddings import EmbeddingFactory
from app.services.retrieval.qdrant_client import QdrantClient
from app.services.retrieval.factory import RetrievalFactory
from app.services.retrieval.vector_base import VectorSearchRequest
from app.api.deps import get_current_user, get_current_active_user
from app.models.database.user import User
from app.config.settings import get_settings


router = APIRouter(prefix="/documents", tags=["文档"])


# 请求/响应模型
class DocumentUploadRequest(BaseModel):
    """文档上传请求（文本方式上传）。"""
    title: str = Field(..., min_length=1, max_length=500, description="文档标题")
    content: str = Field(..., min_length=1, description="文档文本内容")
    metadata: Optional[dict] = Field(default=None, description="可选元数据")


class DocumentUploadResponse(BaseModel):
    """文档上传响应。"""
    document_id: str
    title: str
    chunks_count: int
    total_tokens: int
    embedding_model: str
    chunking_strategy: str
    message: str = "文档上传成功"


class SearchRequest(BaseModel):
    """搜索请求。"""
    query: str = Field(..., min_length=1, description="搜索查询")
    top_k: int = Field(default=5, ge=1, le=20, description="返回结果数量")
    filters: Optional[dict] = Field(default=None, description="可选过滤器")


class SearchResult(BaseModel):
    """单条搜索结果。"""
    chunk_id: str
    document_id: str
    content: str
    score: float
    metadata: Optional[dict] = None


class SearchResponse(BaseModel):
    """搜索响应。"""
    query: str
    results: List[SearchResult]
    total_count: int


class DocumentInfo(BaseModel):
    """文档信息。"""
    document_id: str
    title: str
    chunks_count: int


class DeleteResponse(BaseModel):
    """删除响应。"""
    document_id: str
    deleted_chunks: int
    message: str = "文档删除成功"


# 依赖
async def get_ingestion_service() -> DocumentIngestionService:
    """
    获取文档摄入服务实例。

    使用嵌入与向量库配置创建服务。
    """
    settings = get_settings()

    # 创建嵌入服务
    embedding_service = EmbeddingFactory.create_from_settings()

    # 创建 Qdrant 客户端
    qdrant_client = RetrievalFactory.create_vector_client(
        client_type="qdrant",
        url=settings.VECTOR_DB_URL,
        collection_name=settings.VECTOR_COLLECTION_NAME,
        api_key=settings.VECTOR_API_KEY,
        embedding_service=embedding_service,
    )

    # 创建摄入服务
    return DocumentIngestionService(
        qdrant_client=qdrant_client,
        embedding_provider=settings.EMBEDDING_PROVIDER,
        chunking_strategy="semantic",  # 默认分块策略
    )


async def get_qdrant_client() -> QdrantClient:
    """获取 Qdrant 客户端实例。"""
    settings = get_settings()

    # 创建嵌入服务
    embedding_service = EmbeddingFactory.create_from_settings()

    # 创建 Qdrant 客户端
    return RetrievalFactory.create_vector_client(
        client_type="qdrant",
        url=settings.VECTOR_DB_URL,
        collection_name=settings.VECTOR_COLLECTION_NAME,
        api_key=settings.VECTOR_API_KEY,
        embedding_service=embedding_service,
    )


# 端点
@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="上传文档文本"
)
async def upload_document_text(
    request: DocumentUploadRequest,
    current_user: User = Depends(get_current_active_user),
    ingestion_service: DocumentIngestionService = Depends(get_ingestion_service),
):
    """
    通过文本内容上传文档。

    - **title**: 文档标题
    - **content**: 文档完整文本内容
    - **metadata**: 可选元数据（作者、分类、标签等）

    返回文档 ID 与处理统计信息。
    """
    try:
        # 将用户信息加入元数据（仅存 user_id，不存邮箱等 PII）
        metadata = request.metadata or {}
        metadata["uploaded_by"] = current_user.id

        # 摄入文档
        result = await ingestion_service.ingest_text(
            text=request.content,
            title=request.title,
            metadata=metadata
        )

        return DocumentUploadResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文档上传失败: {str(e)}"
        )


@router.post(
    "/upload/file",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="上传文档文件"
)
async def upload_document_file(
    file: UploadFile = File(...),
    title: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    ingestion_service: DocumentIngestionService = Depends(get_ingestion_service),
):
    """
    上传文档文件（PDF、TXT、MD）。

    支持的格式:
    - PDF（.pdf）
    - 文本（.txt）
    - Markdown（.md）

    文件将被处理、分块并存入向量数据库。
    """
    try:
        # 校验文件类型
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="未提供文件"
            )

        # 检查文件扩展名
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in ['pdf', 'txt', 'md']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"不支持的文件类型: .{file_ext}。支持: pdf、txt、md"
            )

        # 读取文件内容
        content = await file.read()

        # 保存到临时文件供摄入服务使用
        import tempfile
        import os

        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            mode='wb',
            delete=False,
            suffix=f'.{file_ext}'
        ) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        try:
            # 准备元数据（仅存 user_id，不存邮箱等 PII）
            metadata = {
                "uploaded_by": current_user.id,
                "original_filename": file.filename,
            }

            if title:
                metadata["title"] = title

            # 摄入文件
            result = await ingestion_service.ingest_file(
                file_path=temp_path,
                metadata=metadata
            )

            return DocumentUploadResponse(**result)

        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"文件上传失败: {str(e)}"
        )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="搜索文档"
)
async def search_documents(
    request: SearchRequest,
    current_user: User = Depends(get_current_user),
    qdrant_client: QdrantClient = Depends(get_qdrant_client),
):
    """
    使用语义搜索查询相关文档片段。

    - **query**: 搜索查询文本
    - **top_k**: 返回结果数量（1-20）
    - **filters**: 元数据可选过滤器

    返回排序后的相关文档片段列表。
    """
    try:
        # 创建搜索请求
        search_request = VectorSearchRequest(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters
        )

        # 执行搜索
        results = await qdrant_client.search(search_request)

        # 转换为响应格式
        search_results = []
        for result in results:
            search_results.append(SearchResult(
                chunk_id=result.metadata.get("chunk_id", "") if result.metadata else "",
                document_id=result.document_id,
                content=result.content,
                score=result.score,
                metadata=result.metadata
            ))

        return SearchResponse(
            query=request.query,
            results=search_results,
            total_count=len(search_results)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"搜索失败: {str(e)}"
        )


@router.delete(
    "/{document_id}",
    response_model=DeleteResponse,
    summary="删除文档"
)
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_active_user),
    ingestion_service: DocumentIngestionService = Depends(get_ingestion_service),
):
    """
    从向量数据库中删除文档及其全部分块。

    将移除与该文档关联的所有分块。
    """
    try:
        result = await ingestion_service.delete_document(document_id)

        return DeleteResponse(**result)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除文档失败: {str(e)}"
        )


@router.get(
    "/health",
    summary="文档服务健康检查"
)
async def health_check():
    """检查文档服务是否健康。"""
    return {
        "status": "healthy",
        "service": "document-management",
        "embedding_enabled": True,
        "vector_db_enabled": True
    }
