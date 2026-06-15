"""Prometheus metrics + instrumentation (Sprint 14)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import create_app
from app.observability.metrics import PREDICTIONS, record_prediction, render_metrics


def test_render_metrics_exposes_metric_names() -> None:
    text = render_metrics().decode()
    assert "matchoracle_http_requests_total" in text
    assert "matchoracle_predictions_total" in text


def test_record_prediction_increments_counter() -> None:
    before = PREDICTIONS.labels(domain="worldcup")._value.get()
    record_prediction("worldcup")
    after = PREDICTIONS.labels(domain="worldcup")._value.get()
    assert after == before + 1


def test_metrics_endpoint_and_instrumentation() -> None:
    client = TestClient(create_app())
    # Hit a route so the middleware records at least one request.
    assert client.get("/health").status_code == 200

    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]
    body = resp.text
    assert "matchoracle_http_requests_total" in body
    # The /health request we just made is reflected.
    assert "/health" in body
