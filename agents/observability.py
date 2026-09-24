"""In-process metrics: request latency percentiles + per-node LLM token usage.

Deliberately lightweight (no Prometheus/StatsD dependency) -- it's what backs the
``/metrics`` endpoint for local observability. For production this is exactly the
kind of thing you'd instead push through the OpenTelemetry metrics SDK to an OTLP
collector; the shape here mirrors what that would report.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _MetricsStore:
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _latencies_ms: deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    _token_usage: dict[str, dict[str, int]] = field(default_factory=dict)
    request_count: int = 0

    def record_latency(self, duration_ms: float) -> None:
        with self._lock:
            self._latencies_ms.append(duration_ms)
            self.request_count += 1

    def record_llm_usage(self, node: str, prompt_tokens: int, completion_tokens: int) -> None:
        if not prompt_tokens and not completion_tokens:
            return
        with self._lock:
            bucket = self._token_usage.setdefault(
                node, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            )
            bucket["prompt_tokens"] += prompt_tokens
            bucket["completion_tokens"] += completion_tokens
            bucket["total_tokens"] += prompt_tokens + completion_tokens

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latencies = sorted(self._latencies_ms)
            token_usage = {k: dict(v) for k, v in self._token_usage.items()}
            request_count = self.request_count

        def percentile(p: float) -> float | None:
            if not latencies:
                return None
            idx = min(int(len(latencies) * p), len(latencies) - 1)
            return round(latencies[idx], 2)

        return {
            "request_count": request_count,
            "latency_p50_ms": percentile(0.50),
            "latency_p95_ms": percentile(0.95),
            "token_usage_by_node": token_usage,
        }


metrics = _MetricsStore()


def extract_usage(message: Any) -> tuple[int, int]:
    """Best-effort (prompt_tokens, completion_tokens) from a LangChain AIMessage.

    Returns (0, 0) for providers/fake models that don't report usage -- this is
    advisory, not something callers should assume is always populated.
    """
    usage = getattr(message, "usage_metadata", None)
    if usage:
        return usage.get("input_tokens", 0) or 0, usage.get("output_tokens", 0) or 0

    meta = getattr(message, "response_metadata", None) or {}
    token_usage = meta.get("token_usage") or meta.get("usage") or {}
    return token_usage.get("prompt_tokens", 0) or 0, token_usage.get("completion_tokens", 0) or 0
