"""
Custom exception classes for the application.

Provides domain-specific exceptions that can be caught and handled appropriately.
"""


class AppException(Exception):
    """Base exception for all application errors"""

    def __init__(self, message: str, code: str | None = None):
        self.message = message
        self.code = code or "INTERNAL_ERROR"
        super().__init__(self.message)


class ValidationError(AppException):
    """Raised when input validation fails"""

    def __init__(self, message: str):
        super().__init__(message, code="VALIDATION_ERROR")


class AuthenticationError(AppException):
    """Raised when authentication fails"""

    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, code="AUTHENTICATION_FAILED")


class AuthorizationError(AppException):
    """Raised when user is not authorized for an action"""

    def __init__(self, message: str = "Authorization failed"):
        super().__init__(message, code="AUTHORIZATION_FAILED")


class NotFoundError(AppException):
    """Raised when a resource is not found"""

    def __init__(self, resource: str, identifier: str | None = None):
        message = f"{resource} not found"
        if identifier:
            message += f": {identifier}"
        super().__init__(message, code="NOT_FOUND")


class ConflictError(AppException):
    """Raised when a resource already exists"""

    def __init__(self, message: str):
        super().__init__(message, code="CONFLICT")


class RateLimitError(AppException):
    """Raised when rate limit is exceeded"""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, code="RATE_LIMIT_EXCEEDED")


class ExternalServiceError(AppException):
    """Raised when an external service call fails"""

    def __init__(self, service: str, message: str = "External service error"):
        super().__init__(f"{service}: {message}", code="EXTERNAL_SERVICE_ERROR")


class BaseServiceError(AppException):
    """Base exception for service layer errors"""

    def __init__(self, message: str, details: dict | None = None):
        self.details = details or {}
        super().__init__(message, code="SERVICE_ERROR")
