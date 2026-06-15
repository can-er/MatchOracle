"""Observability (Sprint 14): Prometheus metrics + request instrumentation.

Structured logs already flow through structlog (Loki-friendly JSON). This package
adds Prometheus counters/histograms and the ``/metrics`` exposition, gated by
``settings.metrics_enabled``.
"""

from __future__ import annotations

from app.observability.metrics import (
    PREDICTIONS,
    REGISTRY,
    observe_request,
    record_prediction,
    render_metrics,
)

__all__ = [
    "PREDICTIONS",
    "REGISTRY",
    "observe_request",
    "record_prediction",
    "render_metrics",
]
