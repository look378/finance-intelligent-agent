"""Unit tests for prompt hardening."""
from app.services.guardrails.prompt_hardening import build_safe_system_prompt


class TestPromptHardening:
    def test_adds_safety_instructions(self):
        result = build_safe_system_prompt("You are a helpful assistant.")
        assert "You are a helpful assistant." in result
        assert "safety rules" in result
        assert "Never reveal" in result

    def test_preserves_base_prompt(self):
        base = "You are a customer service assistant."
        result = build_safe_system_prompt(base)
        assert base in result
