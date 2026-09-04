"""
Validation utilities.

Custom validators for common validation scenarios.
"""
import re
from typing import Any, Dict


def validate_email(email: str) -> bool:
    """
    Validate email format.

    Args:
        email: Email address to validate

    Returns:
        bool: True if valid, False otherwise
    """
    if not email:
        return False

    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None


def validate_password_strength(password: str) -> Dict[str, Any]:
    """
    Validate password strength.

    Args:
        password: Password to validate

    Returns:
        dict: Dictionary with 'valid' bool and 'errors' list
    """
    errors = []

    if len(password) < 8:
        errors.append("Password must be at least 8 characters long")

    if len(password) > 128:
        errors.append("Password must be less than 128 characters long")

    if not re.search(r"[A-Z]", password):
        errors.append("Password must contain at least one uppercase letter")

    if not re.search(r"[a-z]", password):
        errors.append("Password must contain at least one lowercase letter")

    if not re.search(r"\d", password):
        errors.append("Password must contain at least one digit")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


def validate_message_content(content: str, min_length: int = 1, max_length: int = 5000) -> Dict[str, Any]:
    """
    Validate message content.

    Args:
        content: Message content to validate
        min_length: Minimum length requirement
        max_length: Maximum length requirement

    Returns:
        dict: Dictionary with 'valid' bool and 'errors' list
    """
    errors = []

    if not content:
        errors.append("Message content cannot be empty")
        return {"valid": False, "errors": errors}

    if len(content) < min_length:
        errors.append(f"Message must be at least {min_length} character(s)")

    if len(content) > max_length:
        errors.append(f"Message must be less than {max_length} characters")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


def sanitize_user_input(text: str) -> str:
    """
    Sanitize user input to prevent injection attacks.

    Args:
        text: User input text

    Returns:
        str: Sanitized text
    """
    if not text:
        return ""

    # Remove potential SQL injection patterns
    text = re.sub(r"(-{2}|;|\b(ALTER|CREATE|DELETE|DROP|EXEC|EXECUTE|INSERT|SELECT|UNION|UPDATE)\b)", "", text, flags=re.IGNORECASE)

    # Remove potential script tags
    text = re.sub(r"<script.*?>.*?</script>", "", text, flags=re.IGNORECASE | re.DOTALL)

    # Remove potential event handlers
    text = re.sub(r"on\w+\s*=", "", text, flags=re.IGNORECASE)

    return text.strip()
