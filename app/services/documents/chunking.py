"""
Document chunking strategies.

Implements various strategies for splitting documents into chunks.
"""
import re
from typing import List
import uuid

from app.services.documents.base import ChunkingStrategy, Document, DocumentChunk


class FixedSizeChunking(ChunkingStrategy):
    """
    Fixed-size document chunking strategy.

    Splits documents into chunks of fixed size with overlap.
    Simple and predictable, good for most use cases.
    """

    async def chunk(
        self,
        document: Document,
        max_chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> List[DocumentChunk]:
        """
        Split document into fixed-size chunks.

        Args:
            document: Document to chunk
            max_chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks

        Returns:
            List[DocumentChunk]: List of chunks
        """
        # 参数校验：overlap 必须小于 chunk size，否则 start 不前移会死循环
        if max_chunk_size <= 0:
            max_chunk_size = 512
        if chunk_overlap < 0:
            chunk_overlap = 0
        if chunk_overlap >= max_chunk_size:
            chunk_overlap = max(0, max_chunk_size // 2)

        content = document.content
        chunks = []

        start = 0
        index = 0

        while start < len(content):
            # Calculate end position
            end = start + max_chunk_size

            # If not the last chunk, try to break at word boundary
            if end < len(content):
                # Find last space before end
                last_space = content.rfind(' ', start, end)
                if last_space != -1:
                    end = last_space + 1

            # Extract chunk content
            chunk_content = content[start:end].strip()

            # Skip empty chunks
            if not chunk_content:
                start = end
                continue

            # Create chunk
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document.document_id,
                index=index,
                content=chunk_content,
                metadata={
                    **document.metadata,
                    "title": document.title,
                    "file_type": document.file_type,
                    "start_pos": start,
                    "end_pos": end,
                    "chunk_size": len(chunk_content),
                }
            )
            chunks.append(chunk)

            # Move to next chunk with overlap
            # 防御：保证 start 严格前进，防止 overlap 异常时死循环
            next_start = end - chunk_overlap if end < len(content) else end
            if next_start <= start:
                next_start = end
            start = next_start
            index += 1

        return chunks


class SemanticChunking(ChunkingStrategy):
    """
    Semantic document chunking strategy.

    Splits documents based on semantic boundaries like paragraphs,
    sections, and sentences. Better for maintaining context.
    """

    async def chunk(
        self,
        document: Document,
        max_chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> List[DocumentChunk]:
        """
        Split document into semantic chunks.

        Tries to split at:
        1. Paragraph boundaries
        2. Sentence boundaries
        3. Word boundaries (fallback)

        Args:
            document: Document to chunk
            max_chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks

        Returns:
            List[DocumentChunk]: List of chunks
        """
        content = document.content
        chunks = []

        # First split by paragraphs (double newlines)
        paragraphs = re.split(r'\n\s*\n', content)

        current_chunk = ""
        index = 0
        chunk_start = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If paragraph is small enough, add to current chunk
            if len(current_chunk) + len(para) + 2 <= max_chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # Save current chunk if exists
                if current_chunk:
                    chunk = self._create_chunk(
                        document=document,
                        content=current_chunk,
                        index=index,
                        start=chunk_start,
                    )
                    chunks.append(chunk)
                    index += 1

                    # Update start position for next chunk
                    chunk_start += len(current_chunk)

                # Check if paragraph itself is too large
                if len(para) > max_chunk_size:
                    # Split large paragraph by sentences
                    sentences = self._split_sentences(para)
                    current_chunk = ""

                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                            current_chunk += (" " if current_chunk else "") + sentence
                        else:
                            if current_chunk:
                                chunk = self._create_chunk(
                                    document=document,
                                    content=current_chunk,
                                    index=index,
                                    start=chunk_start,
                                )
                                chunks.append(chunk)
                                index += 1
                                chunk_start += len(current_chunk)
                            current_chunk = sentence
                else:
                    current_chunk = para

        # Add last chunk
        if current_chunk:
            chunk = self._create_chunk(
                document=document,
                content=current_chunk,
                index=index,
                start=chunk_start,
            )
            chunks.append(chunk)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List[str]: List of sentences
        """
        # Simple sentence splitting regex
        # Handles periods, question marks, exclamation marks
        sentences = re.split(r'(?<=[.!?。！？])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _create_chunk(
        self,
        document: Document,
        content: str,
        index: int,
        start: int
    ) -> DocumentChunk:
        """Create a DocumentChunk with proper metadata."""
        return DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document.document_id,
            index=index,
            content=content,
            metadata={
                **document.metadata,
                "title": document.title,
                "file_type": document.file_type,
                "start_pos": start,
                "end_pos": start + len(content),
                "chunk_size": len(content),
                "chunking_strategy": "semantic",
            }
        )


class RecursiveCharacterChunking(ChunkingStrategy):
    """
    Recursive character chunking strategy.

    Tries multiple separators in order to find the best split points:
    1. Paragraph breaks (\n\n)
    2. Sentence breaks (. ! ?)
    3. Word breaks (spaces)
    4. Character breaks (fallback)

    Good for maintaining context while respecting size limits.
    """

    SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "。", "！", "？", " ", ""]

    async def chunk(
        self,
        document: Document,
        max_chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> List[DocumentChunk]:
        """
        Split document using recursive character chunking.

        Args:
            document: Document to chunk
            max_chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between chunks

        Returns:
            List[DocumentChunk]: List of chunks
        """
        chunks = self._recursive_split(
            text=document.content,
            separators=self.SEPARATORS,
            max_size=max_chunk_size,
            overlap=chunk_overlap,
        )

        # Create DocumentChunk objects
        result = []
        position = 0

        for index, chunk_content in enumerate(chunks):
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                document_id=document.document_id,
                index=index,
                content=chunk_content,
                metadata={
                    **document.metadata,
                    "title": document.title,
                    "file_type": document.file_type,
                    "start_pos": position,
                    "end_pos": position + len(chunk_content),
                    "chunk_size": len(chunk_content),
                    "chunking_strategy": "recursive",
                }
            )
            result.append(chunk)
            position += len(chunk_content) - chunk_overlap

        return result

    def _recursive_split(
        self,
        text: str,
        separators: List[str],
        max_size: int,
        overlap: int,
    ) -> List[str]:
        """
        Recursively split text using separators.

        Args:
            text: Text to split
            separators: List of separators to try
            max_size: Maximum chunk size
            overlap: Overlap between chunks

        Returns:
            List[str]: List of text chunks
        """
        # Base case: no separators left, split by character
        if not separators:
            return self._split_by_size(text, max_size, overlap)

        # Try splitting with current separator
        separator = separators[0]
        if separator in text:
            splits = text.split(separator)
        else:
            # Separator not found, try next one
            return self._recursive_split(text, separators[1:], max_size, overlap)

        # Combine splits into chunks
        chunks = []
        current_chunk = ""

        for split in splits:
            if len(current_chunk) + len(separator) + len(split) <= max_size:
                if current_chunk:
                    current_chunk += separator + split
                else:
                    current_chunk = split
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(current_chunk)

                # If split is too large, recurse
                if len(split) > max_size:
                    sub_chunks = self._recursive_split(
                        split,
                        separators[1:],
                        max_size,
                        overlap
                    )
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = split

        # Add last chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_by_size(
        self,
        text: str,
        max_size: int,
        overlap: int
    ) -> List[str]:
        """Split text by character count with overlap."""
        if max_size <= 0:
            max_size = 512
        if overlap < 0:
            overlap = 0
        if overlap >= max_size:
            overlap = max(0, max_size // 2)

        chunks = []
        start = 0

        while start < len(text):
            end = start + max_size
            chunk = text[start:end]
            chunks.append(chunk)
            # 防御：保证 start 严格前进，防止 overlap 异常时死循环
            next_start = end - overlap if end < len(text) else end
            if next_start <= start:
                next_start = end
            start = next_start

        return chunks
