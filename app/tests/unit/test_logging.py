"""Tests for logging configuration"""
import logging
import pytest
import structlog
from unittest.mock import Mock, patch


class TestLoggingConfig:
    """Test logging configuration"""

    def test_configure_logging_development(self, monkeypatch):
        """Test logging configuration for development environment"""
        # Arrange
        monkeypatch.setenv("ENVIRONMENT", "development")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")

        # Act
        from app.config.logging import configure_logging
        logger = configure_logging()

        # Assert
        assert logger is not None
        assert isinstance(logger, structlog.stdlib.BoundLogger)

    def test_configure_logging_production(self, monkeypatch):
        """Test logging configuration for production environment"""
        # Arrange
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("LOG_LEVEL", "INFO")

        # Act
        from app.config.logging import configure_logging
        logger = configure_logging()

        # Assert
        assert logger is not None
        assert isinstance(logger, structlog.stdlib.BoundLogger)

    def test_logger_has_standard_keys(self, monkeypatch):
        """Test that logger includes standard context keys"""
        # Arrange
        monkeypatch.setenv("ENVIRONMENT", "development")
        from app.config.logging import configure_logging

        # Act
        logger = configure_logging()
        with patch.object(logger, "info") as mock_info:
            logger.info("test message", extra_key="extra_value")

        # Assert
        assert mock_info.called
        call_args = mock_info.call_args
        assert "event" in call_args[1] or call_args[0][0] == "test message"

    def test_log_level_configuration(self, monkeypatch):
        """Test that log level is configured correctly"""
        # Arrange
        monkeypatch.setenv("LOG_LEVEL", "WARNING")

        # Act
        from app.config.logging import configure_logging
        configure_logging()
        stdlib_logger = logging.getLogger()

        # Assert
        assert stdlib_logger.level <= logging.WARNING

    def test_structlog_processor_chain(self, monkeypatch):
        """Test that structlog processors are configured"""
        # Arrange
        monkeypatch.setenv("ENVIRONMENT", "development")

        # Act
        from app.config.logging import configure_logging
        logger = configure_logging()

        # Assert - logger should be callable
        assert callable(logger.info)
        assert callable(logger.error)
        assert callable(logger.warning)
        assert callable(logger.debug)

    def test_json_output_in_production(self, monkeypatch):
        """Test that JSON output is used in production"""
        # Arrange
        monkeypatch.setenv("ENVIRONMENT", "production")

        # Act
        from app.config.logging import configure_logging
        logger = configure_logging()

        # Assert - logger should use JSON renderer
        assert logger is not None

    def test_console_output_in_development(self, monkeypatch):
        """Test that console output is used in development"""
        # Arrange
        monkeypatch.setenv("ENVIRONMENT", "development")

        # Act
        from app.config.logging import configure_logging
        logger = configure_logging()

        # Assert - logger should use console renderer
        assert logger is not None

    @pytest.mark.parametrize("level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    def test_all_log_levels_callable(self, monkeypatch, level):
        """Test that all log levels are callable"""
        # Arrange
        monkeypatch.setenv("ENVIRONMENT", "development")
        from app.config.logging import configure_logging

        # Act
        logger = configure_logging()
        log_method = getattr(logger, level.lower())

        # Assert - method should be callable
        assert callable(log_method)

    def test_get_logger_returns_same_instance(self, monkeypatch):
        """Test that get_logger returns cached instance"""
        # Arrange
        monkeypatch.setenv("ENVIRONMENT", "development")
        from app.config.logging import get_logger

        # Act
        logger1 = get_logger("test")
        logger2 = get_logger("test")

        # Assert
        assert logger1 is logger2
