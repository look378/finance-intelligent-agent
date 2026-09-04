"""
Factory for creating intent detector instances.

Provides simple interface for creating intent detectors with proper configuration.
"""
from typing import Optional

from app.services.intent.base import IntentDetector
from app.services.intent.rule_based import RuleBasedIntentDetector
from app.services.intent.llm_based import LLMIntentDetector
from app.services.llm.base import LLMServiceBase


class IntentFactory:
    """
    Factory for creating intent detector instances.

    Provides methods for creating intent detectors with proper dependency injection.
    """

    @staticmethod
    def create(
        detector_type: str = "rule_based",
        llm_service: Optional[LLMServiceBase] = None,
    ) -> IntentDetector:
        """
        Create an intent detector instance.

        Args:
            detector_type: Type of detector ("rule_based", "llm_based", "hybrid")
            llm_service: LLM service (required for llm_based and hybrid detectors)

        Returns:
            IntentDetector: Configured intent detector

        Raises:
            ValueError: If detector_type is invalid or llm_service is missing when required
        """
        if detector_type == "rule_based":
            return RuleBasedIntentDetector()

        elif detector_type == "llm_based":
            if llm_service is None:
                raise ValueError("llm_service is required for llm_based detector")
            return LLMIntentDetector(llm_service=llm_service)

        elif detector_type == "hybrid":
            # Import here to avoid circular dependency
            from app.services.intent.hybrid import HybridIntentDetector
            if llm_service is None:
                raise ValueError("llm_service is required for hybrid detector")
            # Create rule-based and LLM-based detectors
            rule_based = RuleBasedIntentDetector()
            llm_based = LLMIntentDetector(llm_service=llm_service)
            return HybridIntentDetector(rule_based=rule_based, llm_based=llm_based)

        else:
            raise ValueError(f"Unknown detector_type: {detector_type}. Valid options: rule_based, llm_based, hybrid")
