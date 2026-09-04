"""Tests for hybrid intent detector"""
import pytest
from unittest.mock import Mock, AsyncMock
from app.models.enums.intent import Intent
from app.services.intent.base import IntentResult


class TestHybridIntentDetector:
    """Test hybrid intent detector"""

    def test_initialization(self):
        """Test detector initialization with both detectors"""
        # Arrange
        from app.services.intent.hybrid import HybridIntentDetector
        from app.services.intent.rule_based import RuleBasedIntentDetector
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase

        mock_llm = Mock(spec=LLMServiceBase)
        rule_based = RuleBasedIntentDetector()
        llm_based = LLMIntentDetector(llm_service=mock_llm)

        # Act
        detector = HybridIntentDetector(
            rule_based=rule_based,
            llm_based=llm_based,
            confidence_threshold=0.7
        )

        # Assert
        assert detector is not None
        assert detector.rule_based == rule_based
        assert detector.llm_based == llm_based
        assert detector.confidence_threshold == 0.7

    def test_initialization_default_threshold(self):
        """Test detector initialization with default threshold"""
        # Arrange
        from app.services.intent.hybrid import HybridIntentDetector
        from app.services.intent.rule_based import RuleBasedIntentDetector
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase

        mock_llm = Mock(spec=LLMServiceBase)
        rule_based = RuleBasedIntentDetector()
        llm_based = LLMIntentDetector(llm_service=mock_llm)

        # Act
        detector = HybridIntentDetector(
            rule_based=rule_based,
            llm_based=llm_based
        )

        # Assert
        assert detector.confidence_threshold == 0.7  # Default value

    @pytest.mark.asyncio
    async def test_detect_high_confidence_rule_based(self):
        """Test using rule-based when confidence is high"""
        # Arrange
        from app.services.intent.hybrid import HybridIntentDetector
        from app.services.intent.rule_based import RuleBasedIntentDetector

        rule_based = RuleBasedIntentDetector()
        llm_based = Mock()
        llm_based.detect_with_confidence = AsyncMock()

        detector = HybridIntentDetector(
            rule_based=rule_based,
            llm_based=llm_based,
            confidence_threshold=0.7
        )

        # Act - Clear "理财" query should get high confidence from rule-based
        intent = await detector.detect("有什么理财产品推荐")

        # Assert
        assert intent == Intent.WEALTH_QUERY
        # LLM should not be called since rule-based confidence is high
        llm_based.detect_with_confidence.assert_not_called()

    @pytest.mark.asyncio
    async def test_detect_low_confidence_fallback_to_llm(self):
        """Test falling back to LLM when rule-based confidence is low"""
        # Arrange
        from app.services.intent.hybrid import HybridIntentDetector
        from app.services.intent.rule_based import RuleBasedIntentDetector
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase

        mock_llm = Mock(spec=LLMServiceBase)
        rule_based = RuleBasedIntentDetector()
        llm_based = LLMIntentDetector(llm_service=mock_llm)

        # Mock LLM to return specific intent
        llm_based.detect_with_confidence = AsyncMock(
            return_value=IntentResult(
                intent=Intent.FAQ,
                confidence=0.85,
                metadata={"method": "llm"}
            )
        )

        detector = HybridIntentDetector(
            rule_based=rule_based,
            llm_based=llm_based,
            confidence_threshold=0.7
        )

        # Act - Ambiguous query with low rule-based confidence
        intent = await detector.detect("Xylophone zebra yellow")

        # Assert
        # Should use LLM result (FAQ in this mock case)
        assert intent in [Intent.FAQ, Intent.UNKNOWN]
        llm_based.detect_with_confidence.assert_called_once()

    @pytest.mark.asyncio
    async def test_detect_with_confidence_high(self):
        """Test detecting with high confidence returns both results"""
        # Arrange
        from app.services.intent.hybrid import HybridIntentDetector
        from app.services.intent.rule_based import RuleBasedIntentDetector

        rule_based = RuleBasedIntentDetector()
        llm_based = Mock()
        llm_based.detect_with_confidence = AsyncMock()

        detector = HybridIntentDetector(
            rule_based=rule_based,
            llm_based=llm_based,
            confidence_threshold=0.5
        )

        # Act
        result = await detector.detect_with_confidence("有什么理财产品推荐")

        # Assert
        assert result.intent == Intent.WEALTH_QUERY
        assert result.confidence >= 0.5
        # Check metadata indicates which method was used
        if result.metadata:
            assert "method" in result.metadata

    @pytest.mark.asyncio
    async def test_detect_with_confidence_low(self):
        """Test detecting with low confidence from LLM"""
        # Arrange
        from app.services.intent.hybrid import HybridIntentDetector
        from app.services.intent.rule_based import RuleBasedIntentDetector
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase

        mock_llm = Mock(spec=LLMServiceBase)
        rule_based = RuleBasedIntentDetector()
        llm_based = LLMIntentDetector(llm_service=mock_llm)

        # Mock LLM to return low confidence
        llm_based.detect_with_confidence = AsyncMock(
            return_value=IntentResult(
                intent=Intent.UNKNOWN,
                confidence=0.3,
                metadata={"method": "llm"}
            )
        )

        detector = HybridIntentDetector(
            rule_based=rule_based,
            llm_based=llm_based,
            confidence_threshold=0.5
        )

        # Act
        result = await detector.detect_with_confidence("Ambiguous query")

        # Assert
        assert result.confidence < 0.5
        # Should have metadata about which method was used

    @pytest.mark.asyncio
    async def test_detect_with_context(self):
        """Test detection with conversation context"""
        # Arrange
        from app.services.intent.hybrid import HybridIntentDetector
        from app.services.intent.rule_based import RuleBasedIntentDetector

        rule_based = RuleBasedIntentDetector()
        llm_based = Mock()
        llm_based.detect_with_confidence = AsyncMock()

        detector = HybridIntentDetector(
            rule_based=rule_based,
            llm_based=llm_based,
            confidence_threshold=0.7
        )

        context = {
            "previous_messages": [
                {"role": "user", "content": "你好"},
                {"role": "assistant", "content": "您好！有什么可以帮您？"}
            ]
        }

        # Act
        intent = await detector.detect("你好", context=context)

        # Assert
        # Rule-based should detect this as GREETING with high confidence
        assert intent == Intent.GREETING

    @pytest.mark.asyncio
    async def test_priority_rule_based_speed(self):
        """Test that rule-based is tried first for speed"""
        # Arrange
        from app.services.intent.hybrid import HybridIntentDetector
        from app.services.intent.rule_based import RuleBasedIntentDetector

        rule_based = RuleBasedIntentDetector()
        llm_based = Mock()
        llm_based.detect_with_confidence = AsyncMock()

        detector = HybridIntentDetector(
            rule_based=rule_based,
            llm_based=llm_based,
            confidence_threshold=0.7
        )

        # Act - Clear greeting query
        intent = await detector.detect("你好")

        # Assert
        assert intent == Intent.GREETING
        # LLM should not be called for clear queries
        llm_based.detect_with_confidence.assert_not_called()

    def test_custom_confidence_threshold(self):
        """Test custom confidence threshold"""
        # Arrange
        from app.services.intent.hybrid import HybridIntentDetector
        from app.services.intent.rule_based import RuleBasedIntentDetector
        from app.services.intent.llm_based import LLMIntentDetector
        from app.services.llm.base import LLMServiceBase

        mock_llm = Mock(spec=LLMServiceBase)
        rule_based = RuleBasedIntentDetector()
        llm_based = LLMIntentDetector(llm_service=mock_llm)

        # Act - Very low threshold means LLM will be used more
        detector_low = HybridIntentDetector(
            rule_based=rule_based,
            llm_based=llm_based,
            confidence_threshold=0.3
        )

        # Act - Very high threshold means rule-based will be used more
        detector_high = HybridIntentDetector(
            rule_based=rule_based,
            llm_based=llm_based,
            confidence_threshold=0.9
        )

        # Assert
        assert detector_low.confidence_threshold == 0.3
        assert detector_high.confidence_threshold == 0.9
