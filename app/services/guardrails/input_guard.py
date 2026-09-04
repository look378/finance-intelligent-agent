"""
Input guardrail — checks user input before it reaches the LLM.

Detects prompt injection, redacts PII (including financial PII such as
bank cards with Luhn validation), blocks toxic content, and enforces
length limits. All checks are local (no external API dependency).
"""
import re
import logging
from typing import List, Dict, Optional, Pattern

from app.services.guardrails.base import GuardrailResult, InputGuardrail

logger = logging.getLogger(__name__)

# Prompt injection patterns
_INJECTION_PATTERNS: List[Pattern] = [
    re.compile(r"ignore\s+(previous|prior|above|all)\s+(instructions?|prompts?|rules?)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)\s+", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all|your\s+instructions?)", re.IGNORECASE),
    re.compile(r"system\s*:", re.IGNORECASE),
    re.compile(r"<\|im_start\|>", re.IGNORECASE),
    re.compile(r"###\s*instruction", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"dan\s+mode", re.IGNORECASE),
    re.compile(r"developer\s+mode", re.IGNORECASE),
    re.compile(r"override\s+(previous|all|safety)\s*(instructions?|rules?|guidelines?)?", re.IGNORECASE),
]

# PII patterns
_PII_PATTERNS: Dict[str, Pattern] = {
    "phone_cn": re.compile(r"1[3-9]\d{9}"),
    "id_card_cn": re.compile(r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]"),
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "bank_card": re.compile(r"\b(?:62\d{14,17}|4\d{12,18}|5[1-5]\d{14,17}|9\d{15,18})\b"),
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

# Toxicity keywords (Chinese + English)
_TOXICITY_KEYWORDS: List[str] = [
    "炸弹", "制造炸弹", "毒品", "制毒", "自杀", "杀人",
    "恐怖袭击", "袭击计划",
]

BLOCK_MESSAGE = "抱歉，您的输入包含不允许的内容，请重新表述您的问题。"


def _luhn_valid(card_number: str) -> bool:
    """Validate a bank card number using the Luhn algorithm."""
    digits = [int(c) for c in card_number if c.isdigit()]
    if len(digits) < 13:
        return False
    checksum = 0
    double_next = False
    for d in reversed(digits):
        if double_next:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
        double_next = not double_next
    return checksum % 10 == 0


class DefaultInputGuardrail(InputGuardrail):
    """Default input guardrail with injection detection, PII redaction, and toxicity blocking.

    Financial PII (bank cards) is validated with the Luhn algorithm before
    redaction to reduce false positives. Supports two mask styles:
      - "full":    replace with [REDACTED] (default, backward compatible)
      - "partial": keep the last 4 digits visible (e.g. 6222****1234)
    """

    def __init__(
        self,
        enable_pii_redaction: bool = True,
        enable_injection_detection: bool = True,
        enable_toxicity_check: bool = True,
        max_length: int = 5000,
        custom_blocklist: Optional[List[str]] = None,
        pii_mask_style: str = "full",
    ) -> None:
        self._enable_pii = enable_pii_redaction
        self._enable_injection = enable_injection_detection
        self._enable_toxicity = enable_toxicity_check
        self._max_length = max_length
        self._blocklist = _TOXICITY_KEYWORDS + (custom_blocklist or [])
        self._mask_style = pii_mask_style if pii_mask_style in ("full", "partial") else "full"

    def check(self, content: str) -> GuardrailResult:
        if not content or not content.strip():
            return GuardrailResult(
                passed=False, action="block", original_content=content,
                sanitized_content="", violations=["empty_content"],
            )

        violations: List[str] = []
        sanitized = content

        # 1. Length check
        if len(content) > self._max_length:
            violations.append("content_too_long")
            return GuardrailResult(
                passed=False, action="block", original_content=content,
                sanitized_content="", violations=violations,
                metadata={"length": len(content), "max_length": self._max_length},
            )

        # 2. Prompt injection detection
        if self._enable_injection:
            for pattern in _INJECTION_PATTERNS:
                if pattern.search(content):
                    violations.append("prompt_injection_detected")
                    logger.warning("Prompt injection detected: %s", pattern.pattern)
                    return GuardrailResult(
                        passed=False, action="block", original_content=content,
                        sanitized_content=BLOCK_MESSAGE, violations=violations,
                    )

        # 3. Toxicity check
        if self._enable_toxicity:
            content_lower = content.lower()
            for keyword in self._blocklist:
                if keyword in content_lower:
                    violations.append("toxic_content")
                    return GuardrailResult(
                        passed=False, action="block", original_content=content,
                        sanitized_content=BLOCK_MESSAGE, violations=violations,
                    )

        # 4. PII redaction (including financial PII with Luhn validation)
        if self._enable_pii:
            sanitized, pii_types = self._redact_pii(content)
            if pii_types:
                violations.extend(pii_types)

        action = "redact" if violations else "allow"
        return GuardrailResult(
            passed=True, action=action, original_content=content,
            sanitized_content=sanitized, violations=violations,
        )

    def _redact_pii(self, text: str) -> tuple:
        sanitized = text
        found_types: List[str] = []

        for pii_type, pattern in _PII_PATTERNS.items():
            matches = pattern.findall(sanitized)
            if not matches:
                continue

            # Bank cards must pass Luhn validation to avoid redacting arbitrary numbers
            if pii_type == "bank_card":
                valid_matches = []
                for m in matches:
                    if _luhn_valid(m):
                        valid_matches.append(m)
                matches = valid_matches
                if not matches:
                    continue

            found_types.append(f"pii_{pii_type}")
            for match in matches:
                if self._mask_style == "partial":
                    sanitized = sanitized.replace(match, self._partial_mask(match, pii_type))
                else:
                    sanitized = sanitized.replace(match, "[REDACTED]")

        return sanitized, found_types

    @staticmethod
    def _partial_mask(value: str, pii_type: str) -> str:
        """Partial masking that keeps the last 4 chars visible."""
        visible = value[-4:]
        prefix = "*" * max(len(value) - 4, 4)
        return f"{prefix}{visible}"
