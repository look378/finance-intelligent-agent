"""
Text processing utilities.

Common text processing functions for cleaning and normalizing text.
"""
import re
from typing import List, Dict


def clean_text(text: str) -> str:
    """
    Clean and normalize text.

    Removes extra whitespace, normalizes line breaks, and strips leading/trailing whitespace.

    Args:
        text: Text to clean

    Returns:
        str: Cleaned text
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)
    # Normalize line breaks
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length of the text
        suffix: Suffix to add if truncated (default: "...")

    Returns:
        str: Truncated text
    """
    if not text or len(text) <= max_length:
        return text

    # Reserve space for suffix
    suffix_length = len(suffix)
    if max_length <= suffix_length:
        return text[:max_length]

    return text[: max_length - suffix_length] + suffix


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """
    Estimate the number of tokens in text.

    This is a rough estimate. For accurate token counts, use the LLM provider's tokenizer.

    Args:
        text: Text to estimate tokens for
        chars_per_token: Average characters per token (default: 4)

    Returns:
        int: Estimated number of tokens
    """
    if not text:
        return 0
    return len(text) // chars_per_token


def sanitize_html(text: str) -> str:
    """
    Remove HTML tags from text.

    Args:
        text: Text potentially containing HTML

    Returns:
        str: Text with HTML tags removed
    """
    if not text:
        return ""

    # Remove HTML tags
    clean = re.sub(r"<[^>]+>", "", text)
    # Clean up extra whitespace
    clean = re.sub(r"\s+", " ", clean)
    return clean.strip()


def extract_code_blocks(text: str) -> List[Dict[str, str]]:
    """
    Extract code blocks from markdown text.

    Args:
        text: Markdown text potentially containing code blocks

    Returns:
        List[dict]: List of code blocks with 'language' and 'code' keys
    """
    pattern = r"```(\w*)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    return [
        {
            "language": lang if lang else "text",
            "code": code.strip(),
        }
        for lang, code in matches
    ]
