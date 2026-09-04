"""
Document processing services package.

Exports all document-related components including chunking,
preprocessing, and ingestion.
"""
from app.services.documents.base import (
    Document,
    DocumentChunk,
    ChunkingStrategy
)
from app.services.documents.chunking import (
    FixedSizeChunking,
    SemanticChunking,
    RecursiveCharacterChunking
)
from app.services.documents.preprocessing import (
    TextPreprocessor,
    DocumentPreprocessor
)
from app.services.documents.ingestion import DocumentIngestionService

__all__ = [
    # Base classes
    "Document",
    "DocumentChunk",
    "ChunkingStrategy",
    # Chunking strategies
    "FixedSizeChunking",
    "SemanticChunking",
    "RecursiveCharacterChunking",
    # Preprocessing
    "TextPreprocessor",
    "DocumentPreprocessor",
    # Ingestion
    "DocumentIngestionService",
]
