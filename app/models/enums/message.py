"""Message-related enums"""
from enum import Enum


class MessageRole(str, Enum):
    """Role of a message sender"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class MessageStatus(str, Enum):
    """Processing status of a message"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
