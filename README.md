# MatchOracle

**AI-powered, multi-agent prediction platform - flagship mission: the FIFA World Cup 2026.**

MatchOracle predicts football outcomes by combining several specialised *agents*
- each reasoning over a different facet of the problem (history, momentum,
context, risk, market strength, and LLM judgement) - and aggregating their views
into a single **explainable** prediction with a confidence score. The engine
keeps a domain-agnostic core but is currently tuned for the World Cup 2026
(48 teams, 104 matches, group stage → knockout bracket → champion).

It does more than guess a winner: it predicts the **exact scoreline** (Poisson
goals model), **group standings & qualification**, the **knockout bracket**, and
the **tournament champion** - and it **improves itself** from real results with
no human in the loop.

---

## Table of contents
- [Why multi-agent](#why-multi-agent)
- [Architecture](#architecture)
- [The agents](#the-agents)
- [Orchestration & weighting](#orchestration--weighting)
- [The prediction algorithm](#the-prediction-algorithm)
- [Self-improvement (autonomous feedback loop)](#self-improvement-autonomous-feedback-loop)
- [Explainability](#explainability)
- [MCP & connectors](#mcp--connectors)
- [Enterprise features](#enterprise-features)
- [REST API](#rest-api)
- [Tech stack](#tech-stack)
- [Getting started](#getting-started)
- [Deployment](#deployment)
- [Testing](#testing)
- [License](#license)

---

## Why multi-agent

A single model bakes one worldview into one set of weights. MatchOracle instead
runs a **panel of specialists** and resolves their disagreement explicitly. Each
agent emits a normalised `score ∈ [0, 1]` and a `confidence ∈ [0, 1]`; an
orchestration layer weights and fuses them, surfaces conflict, and grounds the
result in a human-readable explanation. Agents can be added, removed, benchmarked
and re-weighted **without touching the graph** - the orchestrator discovers them
from a registry.

## Architecture

```
            ┌─────────────────────────────────────────┐
            │   Data providers                         │
            │   football-data.org · OpenLigaDB · MCP   │
            └───────────────────┬─────────────────────┘
                                │
                       Data collection layer
                                │
            ┌───────────────────▼─────────────────────┐
            │         LangGraph orchestrator           │
            │   (parallel fan-out → expert → fuse)     │
            └──┬─────┬─────┬─────┬─────┬─────┬─────────┘
               ▼     ▼     ▼     ▼     ▼     ▼
          Historical Trend Context Risk Market Expert(LLM)
               └─────┴─────┴──┬──┴─────┴─────┘
                              ▼
              Weighted aggregation + confidence + conflict
                              ▼
                Poisson score model (scoreline, 1X2, bracket)
                              ▼
                   REST API · Dashboard · Cron
```

The backend is a **FastAPI** application. A LangGraph `StateGraph` fans out to
every analytical agent in parallel, then runs the Expert (LLM) agent last so it
can read its peers' outputs. Results are aggregated with a weight vector,
explained, and persisted (predictions + per-agent results) in PostgreSQL.

### Backend components (`app/`)

| Module | Responsibility |
|---|---|
| `app/agents/` | The six agents + `BaseAgent`/`AgentContext` + a registry (auto-discovery) |
| `app/orchestration/` | LangGraph graph, weighted aggregation, explanation, weight manager, end-to-end `PredictionService` |
| `app/prediction/` | Poisson **score** model, **group**/**tournament** Monte-Carlo, **accuracy**, **calibration**, **benchmark**, **feedback** |
| `app/connectors/` | Real data sources (World Cup via football-data.org, OpenLigaDB) behind a `BaseConnector` |
| `app/llm/` | Provider-agnostic LLM wrapper (OpenAI / Ollama / Anthropic) + a cost/quality multi-model router |
| `app/mcp/` | Model Context Protocol client + sources (consumed by the Contextual agent) |
| `app/security/` | JWT auth, RBAC, multi-tenant isolation, audit, secret backend (env/Vault) |
| `app/observability/` | Prometheus metrics + request instrumentation |
| `app/workers/` | Celery app for distributed agent execution |
| `app/api/` | Versioned REST routes (`/api/v1/*`) |
| `app/db/`, `app/repositories/`, `app/schemas/` | SQLAlchemy models, repositories, Pydantic schemas |

## The agents

Each agent reads an `AgentContext` (entity, domain, connectors, optional MCP) and
returns a normalised `AgentResult`.

| Agent | Signal it contributes |
|---|---|
| **Historical** | Long-run relative win rate (from FIFA-ranking strength + head-to-head) |
| **Trend** | Recent momentum / short-term form |
| **Contextual** | Real environmental factors - e.g. **World Cup host-nation advantage** (USA/Canada/Mexico), or a News-Feed signal via MCP |
| **Risk** | Uncertainty / volatility of the matchup, exposes a `risk_level` |
| **Market** | Relative team strength (the "market consensus" proxy) |
| **Expert (LLM)** | Qualitative reasoning over the peers' outputs; falls back to a deterministic, grounded synthesis when no LLM is reachable |

Agents **abstain** rather than inject noise when they have no real signal (e.g.
the Contextual agent stays neutral for a non-host matchup with no news feed).

## Orchestration & weighting

The orchestrator collects every agent result and computes a weighted score:

```
weighted_score = Σ  weightᵢ · scoreᵢ
confidence     = f(agent confidences, agreement)
risk_level     = low | medium | high
```

Default weights (sum ≈ 1.0):

```json
{ "historical": 0.25, "trend": 0.20, "contextual": 0.15,
  "risk": 0.15, "market": 0.10, "expert": 0.15 }
```

Weights are **not static** - see [self-improvement](#self-improvement-autonomous-feedback-loop).
The current vector is cached so workers share one source of truth.

## The prediction algorithm

For a two-team matchup MatchOracle predicts the **final score** with a
**Poisson goals model** - the standard approach in football analytics.

1. **Strength → expected goals (λ).** Each team's strength `s ∈ (0, 1]` comes
   from its FIFA ranking (blended with live form when available). The expected
   goals for each side are:

   ```
   λ_home = base_goals · exp( sensitivity · (s_home − s_away) / 2  + home_adv )
   λ_away = base_goals · exp(−sensitivity · (s_home − s_away) / 2 )
   ```

   with seed constants `base_goals = 1.30`, `sensitivity = 1.8`
   (≈ 2.6 goals/match average, matching World Cup history). At neutral venues
   `home_adv = 0`.

2. **Joint score grid.** Goals for each side are independent Poisson variables.
   MatchOracle builds the `(i, j)` probability grid up to 7–7 and reads off:
   - the **most likely exact scoreline** (the *mode* of the grid),
   - the **outcome probabilities** `P(home) / P(draw) / P(away)`,
   - the **top scorelines** with probabilities.

   > Note: the reported score is the *mode*, not the mean - `mode(Poisson(λ)) =
   > ⌊λ⌋`. With these constants λ tops out around 2.8 even for the biggest
   > mismatch, so the single most probable scoreline is intentionally
   > low-scoring (e.g. `2-0`). This **maximises exact-score hit probability**;
   > blowouts still get the *result* right.

3. **Knockouts.** A regulation draw cannot stand: a winner is forced via an
   extra-time / penalty tie-breaker (`shootout_prob`, a near coin-flip tilted to
   the stronger side), and advancement probabilities are returned.

4. **Whole tournament.** Group standings, qualification, the knockout bracket and
   the champion are estimated by **Monte-Carlo** simulation over all 12 groups
   (`app/prediction/group.py`, `tournament.py`).

5. **Calibration.** The score-model constants can be **fit to real results**
   once enough matches are played (`app/prediction/calibration.py`) - dormant at
   kickoff, then self-tuning.

## Self-improvement (autonomous feedback loop)

MatchOracle closes the loop with **no human in it**:

- **Accuracy tracking** - predicted vs actual: 1X2 accuracy, exact-score
  accuracy, mean goal error, and **Brier score**.
- **Per-agent benchmarking** - directional accuracy, confidence calibration gap,
  contribution, and an `underperforming` flag (worse than a coin flip).
- **Auto-weighting** - measured per-agent accuracy nudges the weight vector with
  **guardrails** (per-agent clamp + bounded learning rate), so a noisy window
  can't collapse the vector. A `rollback` restores the defaults.
- **Autonomous ingestion** - as real World Cup matches finish, results are
  auto-attached to the engine's full-agent predictions and a directional label
  (home win / away win; draws skipped) re-tunes the weights. This runs every
  scheduler cycle during the tournament - the agents that track reality gain
  weight on their own.
- **Human-in-the-loop (optional)** - an `approve / reject / correct` verdict can
  still feed the same guardrailed loop, but it is no longer required.

## Explainability

Every prediction ships with its evidence:

```json
{
  "entity": "France vs Senegal",
  "prediction": "France 2-0 Senegal",
  "confidence": 0.82,
  "risk_level": "medium",
  "contributors": ["historical", "market", "expert"],
  "explanation": "Historical performance and team strength favour France; ...",
  "score_detail": { "p_home_win": 0.77, "p_draw": 0.15, "p_away_win": 0.08, "...": "..." }
}
```

## MCP & connectors

- **MCP** - a sync-friendly Model Context Protocol client. The Contextual agent
  can consume an MCP resource (e.g. a News-Feed server) over stdio via the
  official SDK, with a built-in demo source so it works out of the box.
- **Connectors** - real data behind a `BaseConnector`: the World Cup connector
  (FIFA-ranking strengths + live form/results from **football-data.org**) and an
  OpenLigaDB connector. Adding a source is a new subclass.

## Enterprise features

All gated behind feature flags (off by default), so the open deployment stays
simple:

- **Multi-tenancy & RBAC** - JWT auth, `viewer < analyst < admin` roles, strict
  per-tenant data isolation, audit logging, and a secret backend (env or Vault).
- **Observability** - Prometheus metrics (`/metrics`), request instrumentation,
  structured JSON logs, plus a Grafana dashboard and alert rules in `deploy/`.
- **Distributed execution** - a Celery app fans agent work out to workers over
  Redis; runs in-process (eager) by default.
- **Kubernetes / IaC** - Deployments, Service, CPU HPAs and probes in `deploy/k8s`.

## REST API

Base prefix: `/api/v1`.

| Method & path | Purpose |
|---|---|
| `POST /predict` | Generate an explainable prediction for an entity |
| `GET  /predictions` · `/predictions/{id}` · `/predictions/{id}/agents` | History & per-agent breakdown |
| `GET  /worldcup/matchday/{n}` | Predict every match of a group-stage matchday |
| `GET  /worldcup/groups/{group}` | Group standings + qualification probabilities |
| `GET  /worldcup/champion` | Bracket + champion (full-tournament Monte-Carlo) |
| `GET  /worldcup/accuracy` · `/worldcup/calibration` | Accuracy so far · score-model calibration |
| `POST /predictions/{id}/outcome` | Record a real outcome |
| `GET  /agents/accuracy` · `/agents/benchmark` | Per-agent accuracy & benchmark report |
| `GET  /weights` · `POST /weights/recalculate` · `/weights/auto-tune` · `/weights/rollback` | Inspect / re-tune / autonomous-tune / roll back weights |
| `POST /predictions/{id}/feedback` · `POST /weights/learn-from-feedback` | Human-in-the-loop feedback loop |
| `GET  /mcp/servers` · `GET /llm/router` | MCP sources · multi-model router decision |
| `GET  /health` · `GET /metrics` | Liveness · Prometheus metrics |

Interactive docs at `/docs` (Swagger) when the app is running.

## Tech stack

**Backend:** Python 3.12, FastAPI, Pydantic, SQLAlchemy, Alembic.
**AI/orchestration:** LangGraph, LangChain, OpenAI / Ollama / Anthropic (Claude).
**Data:** PostgreSQL, Redis. **Infra:** Docker, Docker Compose, Kubernetes.
**Observability:** Prometheus, Grafana, structlog. **Distributed:** Celery.
**Serverless target:** Vercel (Python functions + Cron) + Supabase Postgres.

## Getting started

Requires Docker + Docker Compose. The whole stack (app, Postgres, Redis, and a
containerised Ollama for real Expert-agent reasoning) comes up with one command:

```bash
cp .env.example .env          # add a free football-data.org key for live data
docker compose up -d --build  # app on http://localhost:8000  (dashboard at /)
```

Local development (dependencies managed with [uv](https://docs.astral.sh/uv/);
`pyproject.toml` is the source of truth):

```bash
uv sync
uv run alembic upgrade head
uv run uvicorn app.main:app --reload
uv run pytest                 # full test suite
```

## Deployment

- **Docker Compose** - the full stack locally (`docker compose up`). An opt-in
  `enterprise` profile adds Prometheus, Grafana, a Celery worker and Vault.
- **Vercel + Supabase (serverless)** - FastAPI runs as a Vercel Python function
  (`api/index.py`), scheduled by **Vercel Cron**, against **Supabase Postgres**,
  with Claude as the Expert LLM. See `vercel.json`.

## Testing

A comprehensive suite (`tests/`) covers agents, orchestration, the Poisson model,
group/tournament simulation, accuracy/benchmark/feedback, the autonomous loop,
auth/RBAC/tenant isolation, observability and the API surface - all runnable
offline (no network, DB or LLM key required).

```bash
uv run pytest
```

## License

[MIT](LICENSE) © 2026 MatchOracle contributors.

> Football data via [football-data.org](https://www.football-data.org/). This
> project is not affiliated with FIFA. "Most likely scoreline" predictions are
> for entertainment and research - bet responsibly.
