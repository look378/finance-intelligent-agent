"""
Structured logging configuration using structlog.

Provides JSON logging for production and console logging for development.
"""
import logging
import sys
from functools import lru_cache
from typing import Any, List

import structlog
from structlog.types import EventDict, Processor

from app.config.settings import settings


def add_app_context(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """
    Add application-level context to log entries.

    Args:
        logger: Logger instance
        method_name: Method being called
        event_dict: Event dictionary

    Returns:
        EventDict: Updated event dictionary with app context
    """
    event_dict["app"] = "finance-agent"
    event_dict["environment"] = settings.ENVIRONMENT
    return event_dict


def configure_logging() -> structlog.stdlib.BoundLogger:
    """
    Configure structured logging for the application.

    Sets up structlog with appropriate processors for development or production.
    In development: uses console-friendly format with colors
    In production: uses JSON format for log aggregation

    Returns:
        structlog.stdlib.BoundLogger: Configured logger instance
    """
    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL.upper()),
    )

    # Shared processors
    shared_processors: List[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        add_app_context,
    ]

    # Development configuration
    if settings.DEBUG:
        # Console-friendly output with colors
        processors: List[Processor] = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True),
        ]
    else:
        # JSON output for production
        processors = shared_processors + [
            structlog.processors.JSONRenderer(),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger()


@lru_cache
def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a cached logger instance.

    Args:
        name: Optional logger name

    Returns:
        structlog.stdlib.BoundLogger: Logger instance
    """
    return structlog.get_logger(name)


# Configure logging on module import
logger = configure_logging()
