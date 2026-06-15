# Deployment & Observability (Sprint 14)

Enterprise-posture infrastructure for MatchOracle. The default `docker compose
up` stack stays lean; everything here is **opt-in**.

## Local enterprise stack (Docker Compose profile)

Brings up the distributed worker, Prometheus, Grafana and Vault alongside the app:

```bash
docker compose --profile enterprise up -d
```

- App metrics: <http://localhost:8000/metrics>
- Prometheus: <http://localhost:9090>
- Grafana: <http://localhost:3000> (anonymous viewer enabled; admin password `GRAFANA_PASSWORD`)
  - Pre-provisioned datasource + **MatchOracle — Overview** dashboard.
- Vault (dev): <http://localhost:8200> (token `MO_VAULT_TOKEN`)

## Distributed agents

`MO_DISTRIBUTED_AGENTS=true` makes the app fan agent work out to Celery workers
over the Redis broker (`app.workers`). Off by default → Celery runs eagerly
in-process (no broker needed).

## Kubernetes (`deploy/k8s`)

```bash
kubectl apply -f deploy/k8s/namespace.yaml
kubectl -n matchoracle create secret generic matchoracle-secrets \
  --from-env-file=.env
kubectl apply -f deploy/k8s/app.yaml      # Deployment + Service + HPA (CPU 70%)
kubectl apply -f deploy/k8s/worker.yaml   # Celery workers + HPA
```

- Secrets are injected from a `Secret` — never baked into the image (Sprint 13).
- Readiness/liveness probes hit `/health`; HPAs autoscale app (2→10) and workers
  (2→8) on CPU.
- Pods are annotated for Prometheus scraping of `/metrics`.

## Observability signals

- **Metrics** (Prometheus): `matchoracle_http_requests_total`,
  `matchoracle_http_request_duration_seconds`, `matchoracle_predictions_total`.
- **Alerts** (`deploy/prometheus/alerts.yml`): high 5xx rate, p95 latency, app down.
- **Logs**: structured JSON via structlog — ship to Loki with your log agent
  (Promtail/Alloy) for centralised logs alongside Grafana dashboards.
