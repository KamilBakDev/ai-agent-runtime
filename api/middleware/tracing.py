"""Structured logging + OpenTelemetry tracing setup.

Console exporter by default (zero infra needed locally); set
``OTEL_EXPORTER_OTLP_ENDPOINT`` to ship spans to Jaeger/Tempo instead. LangSmith
tracing is opt-in via ``LANGCHAIN_TRACING_V2``/``LANGCHAIN_API_KEY`` and is only
touched here by setting the env vars LangChain itself reads -- its absence never
breaks anything.
"""

from __future__ import annotations

import logging
import os
import time
import uuid

import structlog
from fastapi import FastAPI, Request
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

from agents.config import get_settings
from agents.observability import metrics

_configured = False


def configure_structlog(log_level: str = "INFO") -> None:
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(log_level.upper(), logging.INFO)
            if hasattr(logging, "getLevelNamesMapping")
            else getattr(logging, log_level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def configure_tracing(service_name: str = "ai-agent-runtime") -> trace.Tracer:
    settings = get_settings()
    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: service_name}))

    if settings.otel_exporter_otlp_endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint))
        )
    else:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)

    if settings.langchain_tracing_v2 and settings.langchain_api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.langchain_api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langchain_project)

    return trace.get_tracer(service_name)


def setup_observability(app: FastAPI) -> None:
    global _configured
    settings = get_settings()
    configure_structlog(settings.log_level)
    tracer = configure_tracing()
    logger = structlog.get_logger("api.request")

    @app.middleware("http")
    async def trace_and_log(request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        start = time.perf_counter()
        with tracer.start_as_current_span(f"{request.method} {request.url.path}") as span:
            span.set_attribute("http.method", request.method)
            span.set_attribute("http.route", request.url.path)
            span.set_attribute("request.id", request_id)
            structlog.contextvars.bind_contextvars(request_id=request_id)
            try:
                response = await call_next(request)
            except Exception as exc:
                span.record_exception(exc)
                duration_ms = (time.perf_counter() - start) * 1000
                logger.error(
                    "request.failed",
                    method=request.method,
                    path=request.url.path,
                    duration_ms=round(duration_ms, 2),
                )
                raise
            duration_ms = (time.perf_counter() - start) * 1000
            metrics.record_latency(duration_ms)
            span.set_attribute("http.status_code", response.status_code)
            response.headers["x-request-id"] = request_id
            logger.info(
                "request.completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )
            structlog.contextvars.clear_contextvars()
            return response

    _configured = True
