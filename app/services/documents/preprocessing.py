"""
Document preprocessing utilities.

Handles text cleaning, normalization, and preparation before chunking.
"""
import re
from typing import List, Tuple


class TextPreprocessor:
    """
    Text preprocessing utilities for document content.

    Provides methods for cleaning and normalizing text before
    chunking and embedding.
    """

    @staticmethod
    def clean(text: str) -> str:
        """
        Clean text by removing excessive whitespace and special characters.

        Args:
            text: Text to clean

        Returns:
            str: Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove control characters but keep newlines
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)

        # Normalize quotes（弯引号统一为直引号，修复旧逻辑左引号→右引号的错误）
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        text = text.replace("\u2018", "'").replace("\u2019", "'")

        return text.strip()

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Normalize whitespace in text.

        - Multiple spaces → single space
        - Multiple newlines → double newline
        - Tabs → spaces
        - Leading/trailing whitespace removed

        Args:
            text: Text to normalize

        Returns:
            str: Normalized text
        """
        # Replace tabs with spaces
        text = text.replace('\t', ' ')

        # Normalize multiple spaces to single space (within paragraphs)
        text = re.sub(r' +', ' ', text)

        # Normalize multiple newlines (but keep paragraph breaks)
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove leading/trailing whitespace from each line
        lines = text.split('\n')
        lines = [line.strip() for line in lines]
        text = '\n'.join(lines)

        return text.strip()

    @staticmethod
    def remove_headers_footers(
        text: str,
        header_pattern: str = None,
        footer_pattern: str = None
    ) -> str:
        """
        Remove repeated headers and footers from text.

        Args:
            text: Text to process
            header_pattern: Regex pattern for headers
            footer_pattern: Regex pattern for footers

        Returns:
            str: Text with headers/footers removed
        """
        lines = text.split('\n')
        filtered_lines = []

        # Default patterns: very short lines that appear frequently
        if not header_pattern and not footer_pattern:
            # Simple heuristic: remove lines that are < 20 chars
            # and appear at start/end of many pages
            # This is a simplified approach
            return text

        # Custom patterns
        if header_pattern:
            text = re.sub(header_pattern, '', text)

        if footer_pattern:
            text = re.sub(footer_pattern, '', text)

        return text.strip()

    @staticmethod
    def extract_metadata(text: str) -> dict:
        """
        Extract metadata from text content.

        Attempts to find:
        - Title (first line if it looks like a title)
        - Author (if mentioned)
        - Date (if mentioned)

        Args:
            text: Text to analyze

        Returns:
            dict: Extracted metadata
        """
        metadata = {}
        lines = text.strip().split('\n')

        # First line might be a title
        if lines:
            first_line = lines[0].strip()
            # Title: short, no period at end, might be bold/caps in original
            if len(first_line) < 100 and not first_line.endswith('.'):
                metadata['potential_title'] = first_line

        # Look for patterns like "Author: ..." or "By ..."
        author_patterns = [
            r'Author:\s*(.+)',
            r'By\s+(.+?)(?:\n|$)',
            r'Written by\s+(.+?)(?:\n|$)',
        ]

        for pattern in author_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata['potential_author'] = match.group(1).strip()
                break

        # Look for date patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # ISO format
            r'\d{1,2}/\d{1,2}/\d{4}',  # US format
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}',
        ]

        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                metadata['potential_date'] = match.group(0)
                break

        return metadata

    @staticmethod
    def truncate(
        text: str,
        max_length: int = 100000,
        add_ellipsis: bool = True
    ) -> str:
        """
        Truncate text to maximum length.

        Args:
            text: Text to truncate
            max_length: Maximum length
            add_ellipsis: Whether to add "..." at the end

        Returns:
            str: Truncated text
        """
        if len(text) <= max_length:
            return text

        truncated = text[:max_length]

        if add_ellipsis:
            # Try to truncate at word boundary
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.9:  # If within 90% of max
                truncated = truncated[:last_space]

            truncated += "..."

        return truncated


class DocumentPreprocessor:
    """
    Complete document preprocessing pipeline.

    Applies multiple preprocessing steps in sequence.
    """

    def __init__(
        self,
        clean: bool = True,
        normalize_whitespace: bool = True,
        remove_headers_footers: bool = False,
        extract_metadata: bool = True,
        max_length: int = 100000,
    ):
        """
        Initialize preprocessor.

        Args:
            clean: Apply text cleaning
            normalize_whitespace: Normalize whitespace
            remove_headers_footers: Remove headers/footers
            extract_metadata: Extract metadata
            max_length: Maximum document length
        """
        self.clean = clean
        self.normalize_whitespace = normalize_whitespace
        self.remove_headers_footers = remove_headers_footers
        self.extract_metadata = extract_metadata
        self.max_length = max_length

    async def process(
        self,
        text: str,
        metadata: dict = None
    ) -> Tuple[str, dict]:
        """
        Process text through the preprocessing pipeline.

        Args:
            text: Text to process
            metadata: Existing metadata to merge

        Returns:
            tuple: (processed_text, enhanced_metadata)
        """
        result_text = text
        result_metadata = metadata.copy() if metadata else {}

        # Clean text
        if self.clean:
            result_text = TextPreprocessor.clean(result_text)

        # Normalize whitespace
        if self.normalize_whitespace:
            result_text = TextPreprocessor.normalize_whitespace(result_text)

        # Remove headers/footers
        if self.remove_headers_footers:
            result_text = TextPreprocessor.remove_headers_footers(result_text)

        # Truncate if too long
        result_text = TextPreprocessor.truncate(
            result_text,
            max_length=self.max_length
        )

        # Extract metadata
        if self.extract_metadata:
            extracted = TextPreprocessor.extract_metadata(result_text)
            result_metadata.update(extracted)

        return result_text, result_metadata
