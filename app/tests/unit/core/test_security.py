"""Tests for security utilities"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch


class TestPasswordHashing:
    """Test password hashing and verification"""

    def test_verify_password_correct(self):
        """Test verifying a correct password"""
        # Arrange
        from app.core.security import verify_password
        plain_password = "securepassword123"
        hashed_password = "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYzpLaEmc0W"  # bcrypt hash

        # Act
        is_valid = verify_password(plain_password, hashed_password)

        # Assert
        assert is_valid is True

    def test_verify_password_incorrect(self):
        """Test verifying an incorrect password"""
        # Arrange
        from app.core.security import verify_password
        plain_password = "wrongpassword"
        hashed_password = "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYzpLaEmc0W"

        # Act
        is_valid = verify_password(plain_password, hashed_password)

        # Assert
        assert is_valid is False

    def test_hash_password(self):
        """Test hashing a password"""
        # Arrange
        from app.core.security import hash_password, verify_password
        plain_password = "mypassword123"

        # Act
        hashed = hash_password(plain_password)

        # Assert
        assert hashed is not None
        assert hashed != plain_password
        assert hashed.startswith("$2b$")  # bcrypt hash prefix
        assert verify_password(plain_password, hashed) is True

    def test_hash_password_different_hashes(self):
        """Test that hashing the same password twice produces different hashes"""
        # Arrange
        from app.core.security import hash_password
        password = "samepassword"

        # Act
        hash1 = hash_password(password)
        hash2 = hash_password(password)

        # Assert
        assert hash1 != hash2  # Different due to salt
        # But both should verify correctly
        from app.core.security import verify_password
        assert verify_password(password, hash1) is True
        assert verify_password(password, hash2) is True

    def test_hash_password_empty(self):
        """Test hashing an empty password"""
        # Arrange
        from app.core.security import hash_password
        password = ""

        # Act
        hashed = hash_password(password)

        # Assert
        assert hashed is not None
        assert hashed.startswith("$2b$")


class TestJWTToken:
    """Test JWT token creation and verification"""

    def test_create_access_token(self):
        """Test creating an access token"""
        # Arrange
        from app.core.security import create_access_token
        data = {"sub": "user@example.com"}

        # Act
        token = create_access_token(data)

        # Assert
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0

    def test_create_access_token_with_expiration(self):
        """Test creating an access token with custom expiration"""
        # Arrange
        from app.core.security import create_access_token
        data = {"sub": "user@example.com"}
        expires_delta = timedelta(minutes=30)

        # Act
        token = create_access_token(data, expires_delta)

        # Assert
        assert token is not None

    def test_decode_access_token_valid(self):
        """Test decoding a valid access token"""
        # Arrange
        from app.core.security import create_access_token, decode_access_token
        data = {"sub": "user@example.com", "user_id": 123}
        token = create_access_token(data)

        # Act
        decoded = decode_access_token(token)

        # Assert
        assert decoded is not None
        assert decoded["sub"] == "user@example.com"
        assert decoded["user_id"] == 123
        assert "exp" in decoded

    def test_decode_access_token_invalid(self):
        """Test decoding an invalid access token"""
        # Arrange
        from app.core.security import decode_access_token
        invalid_token = "invalid.token.string"

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            decode_access_token(invalid_token)
        assert "token" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower()

    def test_decode_access_token_expired(self):
        """Test decoding an expired access token"""
        # Arrange
        from app.core.security import create_access_token, decode_access_token
        data = {"sub": "user@example.com"}
        # Create token that's already expired
        expired_delta = timedelta(seconds=-1)
        token = create_access_token(data, expired_delta)

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            decode_access_token(token)
        assert "expired" in str(exc_info.value).lower() or "token" in str(exc_info.value).lower()

    def test_create_refresh_token(self):
        """Test creating a refresh token"""
        # Arrange
        from app.core.security import create_refresh_token
        data = {"sub": "user@example.com"}

        # Act
        token = create_refresh_token(data)

        # Assert
        assert token is not None
        assert isinstance(token, str)


class TestPasswordValidation:
    """Test password validation utilities"""

    def test_validate_password_strong(self):
        """Test validating a strong password"""
        # Arrange
        from app.core.security import validate_password
        password = "StrongP@ssw0rd123"

        # Act
        result = validate_password(password)

        # Assert
        assert result["is_valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_password_too_short(self):
        """Test validating a password that's too short"""
        # Arrange
        from app.core.security import validate_password
        password = "Short1!"

        # Act
        result = validate_password(password)

        # Assert
        assert result["is_valid"] is False
        assert any("at least 8 characters" in err.lower() for err in result["errors"])

    def test_validate_password_no_uppercase(self):
        """Test validating a password without uppercase letters"""
        # Arrange
        from app.core.security import validate_password
        password = "lowercase123!"

        # Act
        result = validate_password(password)

        # Assert
        assert result["is_valid"] is False
        assert any("uppercase" in err.lower() for err in result["errors"])

    def test_validate_password_no_lowercase(self):
        """Test validating a password without lowercase letters"""
        # Arrange
        from app.core.security import validate_password
        password = "UPPERCASE123!"

        # Act
        result = validate_password(password)

        # Assert
        assert result["is_valid"] is False
        assert any("lowercase" in err.lower() for err in result["errors"])

    def test_validate_password_no_digit(self):
        """Test validating a password without digits"""
        # Arrange
        from app.core.security import validate_password
        password = "NoDigits!"

        # Act
        result = validate_password(password)

        # Assert
        assert result["is_valid"] is False
        assert any("digit" in err.lower() for err in result["errors"])

    def test_validate_password_multiple_errors(self):
        """Test validating a password with multiple errors"""
        # Arrange
        from app.core.security import validate_password
        password = "short"  # Too short, no uppercase, no digit, no special char

        # Act
        result = validate_password(password)

        # Assert
        assert result["is_valid"] is False
        assert len(result["errors"]) >= 2
