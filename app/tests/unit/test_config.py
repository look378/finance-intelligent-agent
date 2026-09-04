"""Tests for configuration settings"""
import os
import pytest
from pathlib import Path
from pydantic import ValidationError


class TestSettings:
    """Test application settings"""

    def test_settings_loads_from_env(self, monkeypatch):
        """Test that settings can be loaded from environment variables"""
        # Arrange
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("DEBUG", "false")
        monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://user:pass@localhost:5432/test")
        monkeypatch.setenv("SECRET_KEY", "test-secret-key")

        # Act
        from app.config.settings import settings

        # Assert
        assert settings.ENVIRONMENT == "production"
        assert settings.DEBUG is False
        assert settings.DATABASE_URL == "postgresql+asyncpg://user:pass@localhost:5432/test"
        assert settings.SECRET_KEY == "test-secret-key"

    def test_settings_default_values(self, monkeypatch):
        """Test that settings have sensible defaults"""
        # Arrange - clear env vars
        for key in ["ENVIRONMENT", "DEBUG", "API_PORT"]:
            monkeypatch.delenv(key, raising=False)

        # Act
        from app.config.settings import settings

        # Assert
        assert settings.ENVIRONMENT == "development"
        assert settings.DEBUG is True
        assert settings.API_PORT == 8000

    def test_settings_validation_invalid_database_url(self, monkeypatch):
        """Test that invalid database URL raises validation error"""
        # Arrange
        monkeypatch.setenv("DATABASE_URL", "invalid-url")

        # Act & Assert
        from app.config.settings import settings
        with pytest.raises(ValidationError) as exc_info:
            settings.DATABASE_URL  # Access property to trigger validation
        assert "database" in str(exc_info.value).lower() or "url" in str(exc_info.value).lower()

    def test_settings_validation_missing_secret_key(self, monkeypatch):
        """Test that missing secret key raises validation error"""
        # Arrange
        monkeypatch.delenv("SECRET_KEY", raising=False)

        # Act & Assert
        with pytest.raises(ValidationError) as exc_info:
            from app.config.settings import settings
        assert "secret" in str(exc_info.value).lower() or "key" in str(exc_info.value).lower()

    def test_settings_cors_origins_parsing(self, monkeypatch):
        """Test that CORS origins are parsed correctly"""
        # Arrange
        monkeypatch.setenv("CORS_ORIGINS", '["http://localhost:3000", "http://localhost:8000"]')

        # Act
        from app.config.settings import settings

        # Assert
        assert "http://localhost:3000" in settings.CORS_ORIGINS
        assert "http://localhost:8000" in settings.CORS_ORIGINS

    def test_settings_database_pool_size(self, monkeypatch):
        """Test database pool size settings"""
        # Arrange
        monkeypatch.setenv("DATABASE_POOL_SIZE", "50")
        monkeypatch.setenv("DATABASE_MAX_OVERFLOW", "20")

        # Act
        from app.config.settings import settings

        # Assert
        assert settings.DATABASE_POOL_SIZE == 50
        assert settings.DATABASE_MAX_OVERFLOW == 20

    def test_settings_redis_url(self, monkeypatch):
        """Test Redis URL configuration"""
        # Arrange
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/1")

        # Act
        from app.config.settings import settings

        # Assert
        assert settings.REDIS_URL == "redis://localhost:6379/1"

    def test_settings_openai_configuration(self, monkeypatch):
        """Test OpenAI API configuration"""
        # Arrange
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4")

        # Act
        from app.config.settings import settings

        # Assert
        assert settings.OPENAI_API_KEY == "sk-test-key"
        assert settings.OPENAI_MODEL == "gpt-4"

    def test_settings_jwt_configuration(self, monkeypatch):
        """Test JWT token expiration settings"""
        # Arrange
        monkeypatch.setenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "60")
        monkeypatch.setenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "30")

        # Act
        from app.config.settings import settings

        # Assert
        assert settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES == 60
        assert settings.JWT_REFRESH_TOKEN_EXPIRE_DAYS == 30

    def test_settings_rate_limiting(self, monkeypatch):
        """Test rate limiting configuration"""
        # Arrange
        monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "120")
        monkeypatch.setenv("RATE_LIMIT_PER_HOUR", "2000")

        # Act
        from app.config.settings import settings

        # Assert
        assert settings.RATE_LIMIT_PER_MINUTE == 120
        assert settings.RATE_LIMIT_PER_HOUR == 2000
