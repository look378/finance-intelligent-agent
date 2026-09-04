"""Unit tests for output guardrail."""
import pytest

from app.services.guardrails.output_guard import DefaultOutputGuardrail


class TestOutputGuardrail:
    def setup_method(self):
        self.guard = DefaultOutputGuardrail()

    def test_clean_output_passes(self):
        # 投资相关内容会附加风险提示（合规行为），action 仍为 allow
        result = self.guard.check("该货币基金的管理费率为0.33%。")
        assert result.passed is True
        assert result.action == "allow"
        assert "风险" in result.sanitized_content

    def test_empty_output_passes(self):
        result = self.guard.check("")
        assert result.passed is True

    def test_pii_phone_redacted(self):
        result = self.guard.check("客户电话13812345678，请跟进")
        assert result.was_redacted
        assert "13812345678" not in result.sanitized_content
        assert "[REDACTED]" in result.sanitized_content

    def test_pii_email_redacted(self):
        result = self.guard.check("联系方式: client@bank.com")
        assert result.was_redacted
        assert "client@bank.com" not in result.sanitized_content

    def test_pii_bank_card_redacted(self):
        result = self.guard.check("银行卡号6222021234567890123")
        assert result.was_redacted

    def test_no_pii_passes_through(self):
        content = "该理财产品的起购金额为1000元"
        result = self.guard.check(content)
        assert result.action == "allow"
        assert "起购金额" in result.sanitized_content

    def test_pii_disabled(self):
        guard = DefaultOutputGuardrail(enable_pii_redaction=False)
        result = guard.check("电话13812345678")
        assert not result.was_redacted
        assert "13812345678" in result.sanitized_content

    def test_multiple_pii_types(self):
        result = self.guard.check("电话13812345678，邮箱test@example.com")
        assert result.was_redacted
        assert "pii_phone_cn" in result.violations
        assert "pii_email" in result.violations
