"""
OpenTelemetry tracing setup.

Initializes the OTel tracer provider and instruments FastAPI when
ENABLE_TRACING=True. If tracing is disabled or OTel is not installed,
all operations are no-ops.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

_tracer_provider = None


def setup_tracing(app=None, endpoint: str = "http://jaeger:4317", service_name: str = "finance-agent") -> bool:
    """Initialize OpenTelemetry tracing. Returns True if setup succeeded."""
    global _tracer_provider

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer_provider = provider

        if app is not None:
            try:
                from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
                FastAPIInstrumentor.instrument_app(app)
            except ImportError:
                logger.warning("opentelemetry-instrumentation-fastapi not installed, skipping FastAPI instrumentation")

        logger.info("OpenTelemetry tracing initialized (endpoint=%s)", endpoint)
        return True

    except ImportError:
        logger.info("OpenTelemetry packages not installed, tracing disabled")
        return False
    except Exception as e:
        logger.warning("Failed to initialize tracing: %s", e)
        return False


def get_tracer(name: str = "finance-agent"):
    """Get an OTel tracer, or a no-op tracer if OTel is not configured."""
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except ImportError:
        return _NoOpTracer()


class _NoOpTracer:
    """Fallback tracer when OTel is not installed."""

    def start_as_current_span(self, name, **kwargs):
        return _NoOpSpan()

    def start_span(self, name, **kwargs):
        return _NoOpSpan()


class _NoOpSpan:
    """Fallback span that does nothing."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, key, value):
        pass

    def add_event(self, name, attributes=None):
        pass

    def record_exception(self, exception, attributes=None):
        pass

    def is_recording(self):
        return False
