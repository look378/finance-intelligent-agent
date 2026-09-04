"""Tests for rule-based intent detector (wealth management)"""
import pytest
from app.models.enums.intent import Intent


class TestRuleBasedIntentDetector:
    """Test rule-based intent detector"""

    def test_initialization(self):
        """Test detector initialization"""
        # Arrange & Act
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Assert
        assert detector is not None
        assert hasattr(detector, 'rules')

    def test_detect_wealth_query_keyword(self):
        """Test detecting wealth_query intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("有什么理财产品推荐")

        # Assert
        assert intent == Intent.WEALTH_QUERY

    def test_detect_wealth_query_english(self):
        """Test detecting wealth_query intent from English keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("I want some wealth management products")

        # Assert
        assert intent == Intent.WEALTH_QUERY

    def test_detect_fund_query_keyword(self):
        """Test detecting fund_query intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("查一下基金净值")

        # Assert
        assert intent == Intent.FUND_QUERY

    def test_detect_fund_query_fee(self):
        """Test detecting fund_query intent from fee pattern"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("申购费率是多少")

        # Assert
        assert intent == Intent.FUND_QUERY

    def test_detect_insurance_query_keyword(self):
        """Test detecting insurance_query intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("重疾险多少钱一年")

        # Assert
        assert intent == Intent.INSURANCE_QUERY

    def test_detect_portfolio_query_keyword(self):
        """Test detecting portfolio_query intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("我的持仓收益怎么样")

        # Assert
        assert intent == Intent.PORTFOLIO_QUERY

    def test_detect_risk_assessment_keyword(self):
        """Test detecting risk_assessment intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("我想做一下风险测评")

        # Assert
        assert intent == Intent.RISK_ASSESSMENT

    def test_detect_yield_calc_keyword(self):
        """Test detecting yield_calc intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("10万买一年能赚多少")

        # Assert
        assert intent == Intent.YIELD_CALC

    def test_detect_policy_keyword(self):
        """Test detecting policy intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("赎回规则是什么")

        # Assert
        assert intent == Intent.POLICY

    def test_detect_faq_keyword(self):
        """Test detecting FAQ intent from general question keywords"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act - "能不能" is a FAQ keyword that doesn't match task-oriented patterns
        intent = detector.detect("能不能帮我看看")

        # Assert
        assert intent == Intent.FAQ

    def test_detect_greeting_keyword(self):
        """Test detecting greeting intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("你好")

        # Assert
        assert intent == Intent.GREETING

    def test_detect_greeting_english(self):
        """Test detecting greeting intent from English keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("hello")

        # Assert
        assert intent == Intent.GREETING

    def test_detect_chitchat_keyword(self):
        """Test detecting chitchat intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("哈哈")

        # Assert
        assert intent == Intent.CHITCHAT

    def test_detect_confirm_keyword(self):
        """Test detecting confirm intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("是的")

        # Assert
        assert intent == Intent.CONFIRM

    def test_detect_deny_keyword(self):
        """Test detecting deny intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("不是")

        # Assert
        assert intent == Intent.DENY

    def test_detect_cancel_keyword(self):
        """Test detecting cancel intent from keyword"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("取消")

        # Assert
        assert intent == Intent.CANCEL

    def test_detect_unknown(self):
        """Test detecting unknown intent"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("Xylophone zebra yellow")

        # Assert
        assert intent == Intent.UNKNOWN

    def test_detect_with_confidence_high(self):
        """Test detecting intent with high confidence"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        result = detector.detect_with_confidence("有什么理财产品推荐")

        # Assert
        assert result.intent == Intent.WEALTH_QUERY
        assert result.confidence > 0.5
        assert result.confidence <= 1.0

    def test_detect_with_confidence_low(self):
        """Test detecting intent with low confidence"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        result = detector.detect_with_confidence("Xylophone zebra yellow")

        # Assert
        assert result.intent == Intent.UNKNOWN
        assert result.confidence < 0.5

    def test_detect_with_confidence_metadata(self):
        """Test that confidence result includes metadata"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        result = detector.detect_with_confidence("有什么理财产品推荐")

        # Assert
        assert result.intent == Intent.WEALTH_QUERY
        assert result.confidence > 0
        assert result.metadata is not None
        # Metadata should contain info about matched rules
        assert "matched_rules" in result.metadata

    def test_case_insensitive(self):
        """Test that detection is case-insensitive"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent1 = detector.detect("WEALTH")
        intent2 = detector.detect("wealth")

        # Assert
        assert intent1 == intent2
        assert intent1 == Intent.WEALTH_QUERY

    def test_punctuation_handling(self):
        """Test that punctuation is handled correctly"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("你好！！！")

        # Assert
        assert intent == Intent.GREETING

    def test_empty_query(self):
        """Test handling empty query"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("")

        # Assert
        assert intent == Intent.UNKNOWN

    def test_multiple_keywords_wealth_and_fund(self):
        """Test query with both wealth and fund keywords"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act - Both "理财" and "基金" present
        intent = detector.detect("我要买理财和基金")

        # Assert - Should match one intent based on rule priority
        assert intent in [Intent.WEALTH_QUERY, Intent.FUND_QUERY]

    def test_detect_with_context_parameter(self):
        """Test that detect accepts optional context parameter"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act
        intent = detector.detect("有什么理财产品推荐", context={"previous_messages": []})

        # Assert
        assert intent == Intent.WEALTH_QUERY

    def test_detect_yield_calc_pattern(self):
        """Test detecting yield_calc via regex pattern"""
        # Arrange
        from app.services.intent.rule_based import RuleBasedIntentDetector

        detector = RuleBasedIntentDetector()

        # Act - matches amount × term pattern
        intent = detector.detect("20万放三年能赚多少")

        # Assert
        assert intent == Intent.YIELD_CALC
