"""
Base classes for the guardrails system.

Provides data models and ABCs for input/output content safety checks.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    passed: bool = True
    action: str = "allow"          # "allow", "block", "redact"
    original_content: str = ""
    sanitized_content: str = ""
    violations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def was_blocked(self) -> bool:
        return self.action == "block"

    @property
    def was_redacted(self) -> bool:
        return self.action == "redact"

    @property
    def safe_content(self) -> str:
        return self.sanitized_content if self.sanitized_content else self.original_content


class InputGuardrail(ABC):
    """Abstract base class for input content checks."""

    @abstractmethod
    def check(self, content: str) -> GuardrailResult:
        """Check and optionally sanitize input content."""
        ...


class OutputGuardrail(ABC):
    """Abstract base class for output content checks."""

    @abstractmethod
    def check(self, content: str) -> GuardrailResult:
        """Check and optionally sanitize output content."""
        ...


class GuardrailService:
    """Orchestrates input and output guardrails."""

    def __init__(
        self,
        input_guard: Optional[InputGuardrail] = None,
        output_guard: Optional[OutputGuardrail] = None,
    ) -> None:
        self._input_guard = input_guard
        self._output_guard = output_guard

    def check_input(self, content: str) -> GuardrailResult:
        if self._input_guard is None:
            return GuardrailResult(passed=True, action="allow", original_content=content, sanitized_content=content)
        return self._input_guard.check(content)

    def check_output(self, content: str) -> GuardrailResult:
        if self._output_guard is None:
            return GuardrailResult(passed=True, action="allow", original_content=content, sanitized_content=content)
        return self._output_guard.check(content)
