"""
LLM call tracer for observability.

Wraps LLM generate/generate_stream calls with OpenTelemetry spans,
tracking latency, token counts, model info, and errors.
"""
import time
import logging
from typing import Optional, List, Any
from contextlib import asynccontextmanager

from app.middleware.tracing import get_tracer

logger = logging.getLogger(__name__)


class LLMTracer:
    """Trace LLM calls with OpenTelemetry spans."""

    def __init__(self, service_name: str = "finance-agent.llm") -> None:
        self._tracer = get_tracer(service_name)

    @asynccontextmanager
    async def trace_generate(
        self,
        model: Optional[str] = None,
        intent: Optional[str] = None,
        slot_count: int = 0,
    ):
        """Context manager to trace a single LLM generate call."""
        span = self._tracer.start_as_current_span("chat.generate")
        start_time = time.monotonic()

        try:
            with span as s:
                if hasattr(s, "set_attribute"):
                    if model:
                        s.set_attribute("llm.model", model)
                    if intent:
                        s.set_attribute("chat.intent", intent)
                    s.set_attribute("chat.slot_count", slot_count)

                yield s

                elapsed_ms = (time.monotonic() - start_time) * 1000
                if hasattr(s, "set_attribute"):
                    s.set_attribute("llm.latency_ms", round(elapsed_ms, 2))

        except Exception as e:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            if hasattr(span, "record_exception"):
                span.record_exception(e)
            logger.error("LLM generate failed after %.0fms: %s", elapsed_ms, e)
            raise
