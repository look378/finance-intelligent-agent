"""Graph retrieval services."""
from app.services.graph.retrieval.multi_path_fusion import MultiPathRetrievalFusion
from app.services.graph.retrieval.graph_retrieval_service import GraphRetrievalService

__all__ = ["MultiPathRetrievalFusion", "GraphRetrievalService"]

def __getattr__(name: str):
    if name == "TextToCypherService":
        from app.services.graph.retrieval.text_to_cypher import TextToCypherService
        return TextToCypherService
    if name == "GraphEmbeddingSearch":
        from app.services.graph.retrieval.graph_embedding_search import GraphEmbeddingSearch
        return GraphEmbeddingSearch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
