"""Prometheus metrics (Sprint 14).

A dedicated ``CollectorRegistry`` keeps these metrics isolated from the global
default registry — so building several FastAPI app instances (e.g. in tests)
never raises a duplicate-timeseries error.
"""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

REGISTRY = CollectorRegistry()

HTTP_REQUESTS = Counter(
    "matchoracle_http_requests_total",
    "HTTP requests by method, path and status.",
    labelnames=("method", "path", "status"),
    registry=REGISTRY,
)

HTTP_LATENCY = Histogram(
    "matchoracle_http_request_duration_seconds",
    "HTTP request latency in seconds, by method and path.",
    labelnames=("method", "path"),
    registry=REGISTRY,
)

PREDICTIONS = Counter(
    "matchoracle_predictions_total",
    "Predictions produced, by domain.",
    labelnames=("domain",),
    registry=REGISTRY,
)

CONTENT_TYPE = CONTENT_TYPE_LATEST


def observe_request(method: str, path: str, status: int, duration: float) -> None:
    HTTP_REQUESTS.labels(method=method, path=path, status=str(status)).inc()
    HTTP_LATENCY.labels(method=method, path=path).observe(duration)


def record_prediction(domain: str | None) -> None:
    PREDICTIONS.labels(domain=domain or "generic").inc()


def render_metrics() -> bytes:
    """Prometheus text exposition for the ``/metrics`` endpoint."""
    return generate_latest(REGISTRY)
