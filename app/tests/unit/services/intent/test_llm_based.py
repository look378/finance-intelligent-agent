"""Tests for LLM-based intent detector"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.models.enums.intent import Intent


class TestLLMIntentDetector:
    """Test LLM-based intent detector"""

    def test_initialization(self):
        """Test detector initialization with LLM service"""
        # Arrange
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase

        mock_llm = Mock(spec=LLMServiceBase)

        # Act
        detector = LLMIntentDetector(llm_service=mock_llm)

        # Assert
        assert detector is not None
        assert detector.llm_service == mock_llm

    def test_initialization_default_intents(self):
        """Test detector initialization with default intents"""
        # Arrange
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase

        mock_llm = Mock(spec=LLMServiceBase)

        # Act
        detector = LLMIntentDetector(llm_service=mock_llm)

        # Assert
        assert len(detector.intents) > 0
        assert Intent.WEALTH_QUERY.value in detector.intents
        assert Intent.FAQ.value in detector.intents
        assert Intent.GREETING.value in detector.intents

    @pytest.mark.asyncio
    async def test_detect_wealth_query(self):
        """Test detecting wealth_query intent"""
        # Arrange
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase, LLMResponse

        mock_llm = Mock(spec=LLMServiceBase)
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(content="wealth_query", model="gpt-4")
        )

        detector = LLMIntentDetector(llm_service=mock_llm)

        # Act
        intent = await detector.detect("有什么理财产品推荐")

        # Assert
        assert intent == Intent.WEALTH_QUERY
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_fund_query(self):
        """Test detecting fund_query intent"""
        # Arrange
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase, LLMResponse

        mock_llm = Mock(spec=LLMServiceBase)
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(content="fund_query", model="gpt-4")
        )

        detector = LLMIntentDetector(llm_service=mock_llm)

        # Act
        intent = await detector.detect("查一下基金净值")

        # Assert
        assert intent == Intent.FUND_QUERY

    @pytest.mark.asyncio
    async def test_detect_with_confidence(self):
        """Test detecting with confidence score"""
        # Arrange
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase, LLMResponse

        # Return JSON with confidence
        mock_llm = Mock(spec=LLMServiceBase)
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(
                content='{"intent": "wealth_query", "confidence": 0.95}',
                model="gpt-4"
            )
        )

        detector = LLMIntentDetector(llm_service=mock_llm)

        # Act
        result = await detector.detect_with_confidence("有什么理财产品推荐")

        # Assert
        assert result.intent == Intent.WEALTH_QUERY
        assert result.confidence == 0.95

    @pytest.mark.asyncio
    async def test_detect_invalid_json_response(self):
        """Test handling invalid JSON from LLM"""
        # Arrange
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase, LLMResponse

        mock_llm = Mock(spec=LLMServiceBase)
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(content="just plain text", model="gpt-4")
        )

        detector = LLMIntentDetector(llm_service=mock_llm)

        # Act
        intent = await detector.detect("你好")

        # Assert - plain text doesn't match any intent value, so should be UNKNOWN
        assert intent == Intent.UNKNOWN

    @pytest.mark.asyncio
    async def test_detect_with_context(self):
        """Test detection with conversation context"""
        # Arrange
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase, LLMResponse

        mock_llm = Mock(spec=LLMServiceBase)
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(content="chitchat", model="gpt-4")
        )

        detector = LLMIntentDetector(llm_service=mock_llm)

        context = {
            "previous_messages": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "您好！有什么可以帮您？"}
            ]
        }

        # Act
        intent = await detector.detect("哈哈，谢谢", context=context)

        # Assert
        assert intent == Intent.CHITCHAT

    @pytest.mark.asyncio
    async def test_detect_llm_error(self):
        """Test handling LLM API errors"""
        # Arrange
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase
        from app.core.exceptions import ExternalServiceError

        mock_llm = Mock(spec=LLMServiceBase)
        mock_llm.generate = AsyncMock(
            side_effect=Exception("API Error")
        )

        detector = LLMIntentDetector(llm_service=mock_llm)

        # Act & Assert
        with pytest.raises(ExternalServiceError):
            await detector.detect("测试查询")

    def test_custom_prompt_template(self):
        """Test using custom prompt template"""
        # Arrange
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase

        mock_llm = Mock(spec=LLMServiceBase)

        custom_prompt = "Classify this: {query}\nIntents: {intents}"

        # Act
        detector = LLMIntentDetector(
            llm_service=mock_llm,
            prompt_template=custom_prompt
        )

        # Assert
        assert detector.prompt_template == custom_prompt

    def test_build_system_prompt(self):
        """Test system prompt building"""
        # Arrange
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase

        mock_llm = Mock(spec=LLMServiceBase)
        detector = LLMIntentDetector(llm_service=mock_llm)

        # Act
        prompt = detector._build_system_prompt()

        # Assert
        assert "intent" in prompt.lower()
        assert "classify" in prompt.lower()

    def test_build_user_prompt(self):
        """Test user prompt building with query"""
        # Arrange
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase

        mock_llm = Mock(spec=LLMServiceBase)
        detector = LLMIntentDetector(llm_service=mock_llm)

        # Act
        prompt = detector._build_user_prompt("有什么理财产品推荐", context=None)

        # Assert
        assert "有什么理财产品推荐" in prompt
        assert "Intent:" in prompt
