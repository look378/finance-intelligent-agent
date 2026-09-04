"""Entity and relationship extraction from text and structured data."""
from app.services.graph.extraction.base import EntityExtractor, ExtractionResult

__all__ = ["EntityExtractor", "ExtractionResult"]

def __getattr__(name: str):
    if name == "LLMEntityExtractor":
        from app.services.graph.extraction.llm_extractor import LLMEntityExtractor
        return LLMEntityExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
