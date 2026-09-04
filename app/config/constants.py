"""
Application constants.

Central place for all constant values used throughout the application.
"""
from datetime import timedelta

# API Constants
API_V1_PREFIX = "/api/v1"

# Session Constants
DEFAULT_SESSION_TITLE = "New Chat"
DEFAULT_CONTEXT_WINDOW = 10  # Number of messages to keep in context
MAX_CONTEXT_WINDOW = 100
MIN_CONTEXT_WINDOW = 1

# Memory Types
MEMORY_SLIDING_WINDOW = "sliding_window"
MEMORY_SUMMARIZATION = "summarization"
MEMORY_HYBRID = "hybrid"

MEMORY_TYPES = [MEMORY_SLIDING_WINDOW, MEMORY_SUMMARIZATION, MEMORY_HYBRID]

# Message Constants
class MessageRole:
    """Message roles"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageStatus:
    """Message processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


MAX_MESSAGE_LENGTH = 5000
MIN_MESSAGE_LENGTH = 1

# Token Constants
MAX_TOKENS_DEFAULT = 2000
MIN_TOKENS = 1
ESTIMATED_CHARS_PER_TOKEN = 4

# JWT Constants
JWT_ACCESS_TOKEN_EXPIRE = timedelta(minutes=30)
JWT_REFRESH_TOKEN_EXPIRE = timedelta(days=7)

# Pagination Constants
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
MIN_PAGE_SIZE = 1

# Rate Limiting Constants
RATE_LIMIT_DEFAULT = "60/minute"
RATE_LIMIT_BURST = "100/minute"

# Document Constants
MAX_DOCUMENT_SIZE_MB = 50
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50

# Cache Keys
CACHE_KEY_SESSION = "session:{session_id}"
CACHE_KEY_USER_SESSIONS = "user:{user_id}:sessions"
CACHE_KEY_DOCUMENT = "document:{doc_id}"

# HTTP Status Codes (custom messages)
STATUS_MESSAGES = {
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    429: "Too Many Requests",
    500: "Internal Server Error",
    503: "Service Unavailable",
}

# Error Codes
class ErrorCode:
    """Standard error codes"""
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    AUTHORIZATION_FAILED = "AUTHORIZATION_FAILED"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    MESSAGE_TOO_LONG = "MESSAGE_TOO_LONG"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    LLM_API_ERROR = "LLM_API_ERROR"
    VECTOR_DB_ERROR = "VECTOR_DB_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
