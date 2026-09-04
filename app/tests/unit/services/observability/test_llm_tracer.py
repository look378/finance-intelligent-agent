"""Unit tests for LLM tracer (no OTel dependency required)."""
import pytest

from app.services.observability.llm_tracer import LLMTracer


class TestLLMTracer:
    def setup_method(self):
        self.tracer = LLMTracer()

    @pytest.mark.asyncio
    async def test_trace_generate_succeeds(self):
        async with self.tracer.trace_generate(model="test-model", intent="question") as span:
            pass  # simulate LLM call

    @pytest.mark.asyncio
    async def test_trace_generate_with_exception(self):
        with pytest.raises(ValueError):
            async with self.tracer.trace_generate() as span:
                raise ValueError("test error")

    @pytest.mark.asyncio
    async def test_trace_generate_with_slots(self):
        async with self.tracer.trace_generate(slot_count=3) as span:
            pass
