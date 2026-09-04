"""
Hybrid intent detector combining rule-based and LLM approaches.

Uses fast rule-based detection when confident, falls back to LLM for complex queries.
"""
from typing import Optional, Dict

from app.services.intent.base import IntentDetector, IntentResult
from app.services.intent.rule_based import RuleBasedIntentDetector
from app.services.intent.llm_based import LLMIntentDetector
from app.models.enums.intent import Intent


class HybridIntentDetector(IntentDetector):
    """
    Hybrid intent detector combining rule-based and LLM approaches.

    Strategy:
    1. Try rule-based detection first (fast, cheap)
    2. If confidence is below threshold, use LLM (accurate, expensive)
    3. Balance between speed and accuracy
    """

    def __init__(
        self,
        rule_based: RuleBasedIntentDetector,
        llm_based: LLMIntentDetector,
        confidence_threshold: float = 0.7,
    ) -> None:
        """
        Initialize the hybrid intent detector.

        Args:
            rule_based: Rule-based detector instance
            llm_based: LLM-based detector instance
            confidence_threshold: Threshold for using LLM fallback (0.0-1.0)
        """
        self.rule_based = rule_based
        self.llm_based = llm_based
        self.confidence_threshold = confidence_threshold

    async def detect(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> Intent:
        """
        Detect intent using hybrid approach.

        Args:
            query: User's text query
            context: Optional context

        Returns:
            Intent: Detected intent category
        """
        result = await self.detect_with_confidence(query, context)
        return result.intent

    async def detect_with_confidence(
        self,
        query: str,
        context: Optional[dict] = None,
    ) -> IntentResult:
        """
        Detect intent with confidence using hybrid approach.

        Args:
            query: User's text query
            context: Optional context

        Returns:
            IntentResult: Detected intent with confidence and metadata
        """
        # Try rule-based first
        rule_result = self.rule_based.detect_with_confidence(query, context)

        # Use rule-based result if confidence is high enough
        if rule_result.confidence >= self.confidence_threshold:
            return IntentResult(
                intent=rule_result.intent,
                confidence=rule_result.confidence,
                metadata={
                    "method": "rule_based",
                    "threshold": self.confidence_threshold,
                    **(rule_result.metadata or {})
                },
            )

        # Fall back to LLM for low confidence
        llm_result = await self.llm_based.detect_with_confidence(query, context)

        return IntentResult(
            intent=llm_result.intent,
            confidence=llm_result.confidence,
            metadata={
                "method": "llm_fallback",
                "rule_confidence": rule_result.confidence,
                "threshold": self.confidence_threshold,
                "rule_intent": rule_result.intent.value,
                **(llm_result.metadata or {})
            },
        )
