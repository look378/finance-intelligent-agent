"""
Output guardrail — checks LLM output before returning to the user.

Redacts PII that the LLM may have leaked, detects non-compliant financial
marketing language (return promises), and appends mandatory risk
disclaimers for investment-related content.
"""
import re
import logging
from typing import List, Dict, Optional, Pattern

from app.services.guardrails.base import GuardrailResult, OutputGuardrail

logger = logging.getLogger(__name__)

# Reuse PII patterns from input guard
_PII_PATTERNS: Dict[str, Pattern] = {
    "phone_cn": re.compile(r"1[3-9]\d{9}"),
    "id_card_cn": re.compile(r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]"),
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "bank_card": re.compile(r"6[0-9]{15,18}"),
}

FALLBACK_RESPONSE = "抱歉，我无法回答这个问题。"

# Non-compliant marketing language — promising guaranteed returns
_COMPLIANCE_VIOLATION_PATTERNS: List[Pattern] = [
    re.compile(r"保本|稳赚|稳赚不赔|保证收益|无风险|零风险|100%[收获]益|必赚|包赚", re.IGNORECASE),
    re.compile(r"收益[^，。；,\s]*保证|保证[^，。；,\s]*收益", re.IGNORECASE),
    re.compile(r"稳赚不赔|只赚不亏|稳赚", re.IGNORECASE),
]

# Investment-related keywords → risk disclaimer required
_INVESTMENT_KEYWORDS = [
    "理财", "基金", "保险", "收益", "投资", "净值", "赎回", "申购",
    "wealth", "fund", "investment", "return", "yield",
]

RISK_DISCLAIMER = (
    "理财非存款，产品有风险，投资须谨慎。以上内容仅供参考，不构成投资建议，"
    "实际收益以产品净值和合同条款为准。"
)

COMPLIANCE_BLOCK_MESSAGE = (
    "抱歉，根据金融营销宣传合规要求，我不能承诺或暗示保证收益。"
    "理财非存款，产品有风险，投资须谨慎。如需了解具体产品，请提供产品名称或代码。"
)


class DefaultOutputGuardrail(OutputGuardrail):
    """Default output guardrail with PII redaction and compliance checks."""

    def __init__(
        self,
        enable_pii_redaction: bool = True,
        enable_compliance_check: bool = True,
        append_risk_disclaimer: bool = True,
    ) -> None:
        self._enable_pii = enable_pii_redaction
        self._enable_compliance = enable_compliance_check
        self._append_disclaimer = append_risk_disclaimer

    def check(self, content: str) -> GuardrailResult:
        if not content:
            return GuardrailResult(
                passed=True, action="allow", original_content=content, sanitized_content=content,
            )

        violations: List[str] = []
        sanitized = content

        # PII redaction on output
        if self._enable_pii:
            sanitized, pii_types = self._redact_pii(content)
            violations.extend(pii_types)

        # Compliance: block guaranteed-return promises
        if self._enable_compliance:
            for pattern in _COMPLIANCE_VIOLATION_PATTERNS:
                if pattern.search(sanitized):
                    violations.append("compliance_return_promise")
                    logger.warning("Output contains return promise: %s", pattern.pattern)
                    return GuardrailResult(
                        passed=False,
                        action="block",
                        original_content=content,
                        sanitized_content=COMPLIANCE_BLOCK_MESSAGE,
                        violations=violations,
                        metadata={"violation_type": "return_promise"},
                    )

            # Append risk disclaimer for investment-related content
            if self._append_disclaimer and self._is_investment_related(sanitized):
                if RISK_DISCLAIMER not in sanitized:
                    sanitized = f"{sanitized}\n\n{RISK_DISCLAIMER}"
                    violations.append("risk_disclaimer_appended")

        # 只有 PII 脱敏算 redact；追加风险提示不影响 allow 语义
        has_pii = any(v.startswith("pii_") for v in violations)
        action = "redact" if has_pii else "allow"
        return GuardrailResult(
            passed=True, action=action, original_content=content,
            sanitized_content=sanitized, violations=violations,
        )

    def _is_investment_related(self, content: str) -> bool:
        lowered = content.lower()
        return any(kw.lower() in lowered for kw in _INVESTMENT_KEYWORDS)

    @staticmethod
    def _redact_pii(text: str) -> tuple:
        sanitized = text
        found_types: List[str] = []
        for pii_type, pattern in _PII_PATTERNS.items():
            matches = pattern.findall(sanitized)
            if matches:
                found_types.append(f"pii_{pii_type}")
                sanitized = pattern.sub("[REDACTED]", sanitized)
        return sanitized, found_types
