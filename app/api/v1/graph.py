"""
图谱 API 端点（GraphRAG 操作）。

提供图谱查询、schema 检查、社区管理、结构化数据导入与健康检查的 REST API。
"""
import logging
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from pydantic import BaseModel, Field

from app.api.deps import get_current_active_user
from app.models.database.user import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/graph", tags=["图谱"])

_graph_client = None


def set_graph_client(client):
    """设置图谱客户端实例（启动时调用）。"""
    global _graph_client
    _graph_client = client


def _get_client():
    if _graph_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="图谱服务未初始化。请设置 GRAPH_RAG_ENABLED=true。",
        )
    return _graph_client


# --- 模型 ---


class GraphQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="自然语言查询")
    max_results: int = Field(20, gt=0, le=100)


class GraphQueryResponse(BaseModel):
    results: List[dict]
    count: int


class StructuredImportRequest(BaseModel):
    csv_content: str = Field(..., min_length=1)
    entity_mappings: List[dict]
    relation_mappings: Optional[List[dict]] = None
    delimiter: str = Field(",", max_length=1)


# --- 端点 ---


@router.get("/health", summary="图谱服务健康检查")
async def graph_health(
    current_user: User = Depends(get_current_active_user),
):
    """检查图谱数据库连通性。"""
    try:
        client = _get_client()
        healthy = await client.health_check()
        return {"status": "healthy" if healthy else "unhealthy"}
    except HTTPException:
        return {"status": "disabled"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.get("/stats", summary="图谱统计信息")
async def graph_stats(
    current_user: User = Depends(get_current_active_user),
):
    """返回实体与关系数量统计。"""
    client = _get_client()
    try:
        stats = await client.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schema", summary="获取图谱 schema")
async def graph_schema(
    current_user: User = Depends(get_current_active_user),
):
    """返回当前知识图谱 schema。"""
    client = _get_client()
    try:
        schema = await client.get_schema()
        return {"schema": schema}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=GraphQueryResponse, summary="自然语言图谱查询")
async def graph_query(
    request: GraphQueryRequest,
    current_user: User = Depends(get_current_active_user),
):
    """通过 Text-to-Cypher 执行自然语言查询（仅只读）。"""
    from app.services.graph.retrieval import TextToCypherService
    from app.services.llm import LLMFactory
    from app.config.settings import get_settings

    client = _get_client()
    settings = get_settings()

    try:
        llm_service = LLMFactory.create_from_settings()
        t2c = TextToCypherService(
            llm_service=llm_service,
            graph_client=client,
            max_results=request.max_results,
        )
        results = await t2c.query(request.query, max_results=request.max_results)
        return GraphQueryResponse(
            results=[
                {
                    "content": r.content,
                    "score": r.score,
                    "entities": [{"name": e.name, "type": e.type} for e in r.entities],
                    "source_type": r.source_type,
                }
                for r in results
            ],
            count=len(results),
        )
    except Exception as e:
        logger.error("Graph query failed: %s", e)
        raise HTTPException(status_code=500, detail="图谱查询失败")


@router.post("/import/structured", summary="导入结构化数据")
async def import_structured(
    request: StructuredImportRequest,
    current_user: User = Depends(get_current_active_user),
):
    """将结构化 CSV 数据导入知识图谱（写操作，需管理员）。"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限",
        )
    from app.services.graph.extraction.structured_importer import (
        StructuredDataImporter,
        ColumnMapping,
        RelationMapping,
    )

    client = _get_client()

    try:
        importer = StructuredDataImporter()
        entity_mappings = [
            ColumnMapping(
                entity_type=m["entity_type"],
                name_column=m["name_column"],
                property_columns=m.get("property_columns"),
                description_column=m.get("description_column"),
            )
            for m in request.entity_mappings
        ]

        relation_mappings = None
        if request.relation_mappings:
            relation_mappings = [
                RelationMapping(
                    source_entity_type=r["source_entity_type"],
                    source_name_column=r["source_name_column"],
                    target_entity_type=r["target_entity_type"],
                    target_name_column=r["target_name_column"],
                    relation_type=r["relation_type"],
                    relation_properties=r.get("relation_properties"),
                )
                for r in request.relation_mappings
            ]

        entities, relations = importer.import_csv(
            request.csv_content,
            entity_mappings,
            relation_mappings,
            delimiter=request.delimiter,
        )

        entity_ids = await client.add_entities(entities) if entities else []
        relation_ids = await client.add_relations(relations) if relations else []

        return {
            "entities_imported": len(entity_ids),
            "relations_imported": len(relation_ids),
        }
    except Exception as e:
        logger.error("Structured import failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/communities/detect", summary="社区发现检测")
async def detect_communities(
    min_size: int = Query(3, gt=0, description="最小社区规模"),
    max_levels: int = Query(5, gt=0, description="最大层级数"),
    current_user: User = Depends(get_current_active_user),
):
    """对知识图谱执行社区发现（写操作，需管理员）。"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限",
        )
    from app.services.graph.community import CommunityDetectionService

    client = _get_client()

    try:
        service = CommunityDetectionService(client)
        communities = await service.detect_communities(
            min_community_size=min_size,
            max_levels=max_levels,
        )

        total = sum(len(comms) for comms in communities.values())
        return {
            "levels": len(communities),
            "total_communities": total,
            "details": {
                str(level): len(comms) for level, comms in communities.items()
            },
        }
    except Exception as e:
        logger.error("Community detection failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
