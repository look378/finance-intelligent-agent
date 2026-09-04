"""
Base document processing interfaces and data structures.
"""
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class DocumentChunk:
    """
    A chunk of text from a document.

    Attributes:
        content: Text content of the chunk
        chunk_id: Unique identifier for the chunk
        document_id: ID of the source document
        index: Position of chunk in the document
        metadata: Additional metadata (title, page, etc.)
    """
    content: str
    chunk_id: str
    document_id: str
    index: int
    metadata: dict


@dataclass
class Document:
    """
    A document with its content and metadata.

    Attributes:
        document_id: Unique identifier
        title: Document title
        content: Full text content
        file_type: Type of file (pdf, txt, md, etc.)
        metadata: Additional metadata
    """
    document_id: str
    title: str
    content: str
    file_type: str
    metadata: dict


class ChunkingStrategy(ABC):
    """
    Abstract base class for document chunking strategies.

    Different strategies can be used to split documents into chunks:
    - Fixed size chunking
    - Semantic chunking
    - Recursive character chunking
    """

    @abstractmethod
    async def chunk(
        self,
        document: Document,
        max_chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> List[DocumentChunk]:
        """
        Split a document into chunks.

        Args:
            document: Document to chunk
            max_chunk_size: Maximum size of each chunk (in characters/tokens)
            chunk_overlap: Overlap between consecutive chunks

        Returns:
            List[DocumentChunk]: List of document chunks

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("chunk() must be implemented by subclass")
