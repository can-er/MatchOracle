"""End-to-end prediction service (Sprints 05 & 06).

Single entry point the API layer calls. It ties the orchestration graph to the
aggregation, explanation and persistence layers:

    run agents (LangGraph) → aggregate (weights, confidence, conflict)
    → ground the explanation → persist Prediction + AgentResult rows.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

import app.agents  # noqa: F401 — import side-effect registers all built-in agents
from app.config import settings
from app.db.models import AccuracySnapshot as AccuracySnapshotModel
from app.db.models import AgentResult as AgentResultModel
from app.db.models import Feedback as FeedbackModel
from app.db.models import Outcome as OutcomeModel
from app.db.models import Prediction as PredictionModel
from app.logging_config import get_logger
from app.observability.metrics import record_prediction
from app.orchestration.aggregation import aggregate
from app.orchestration.explanation import build_explanation
from app.orchestration.graph import build_graph
from app.orchestration.weights import WeightManager, get_weight_manager
from app.prediction.accuracy import AccuracyReport, evaluate
from app.prediction.benchmark import (
    AgentBenchmark,
    agent_accuracy,
    benchmark_agents,
    realised_label,
)
from app.prediction.calibration import Calibration, calibrate
from app.prediction.feedback import feedback_reward, label_from_feedback
from app.prediction.group import GroupPrediction, simulate_group
from app.prediction.params import set_params
from app.prediction.score import predict_scoreline
from app.prediction.tournament import TournamentPrediction, simulate_tournament
from app.repositories.prediction_repository import (
    AgentResultRepository,
    FeedbackRepository,
    OutcomeRepository,
    PredictionRepository,
)
from app.schemas.agent import AgentResult
from app.schemas.prediction import PredictionRequest

logger = get_logger(__name__)


def _parse_kickoff(utc: str | None) -> datetime | None:
    if not utc:
        return None
    try:
        return datetime.fromisoformat(utc.replace("Z", "+00:00"))
    except ValueError:
        return None


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


class PredictionService:
    """Run a full prediction for one entity and persist the result."""

    def __init__(self, session: Session, *, weight_manager: WeightManager | None = None) -> None:
        self.session = session
        self.weights = weight_manager or get_weight_manager()
        self.predictions = PredictionRepository(session)
        self.agent_results = AgentResultRepository(session)
        self.outcomes = OutcomeRepository(session)
        self.feedback = FeedbackRepository(session)

    def predict(
        self, request: PredictionRequest, *, tenant_id: str | None = None
    ) -> PredictionModel:
        """Execute the agents, aggregate, explain and persist a prediction."""
        results = self._run_agents(request)
        weights = self.weights.current()
        agg = aggregate(results, weights)
        explanation = build_explanation(agg, results)
        scoreline = self._scoreline_for(request)

        prediction = PredictionModel(
            entity=request.entity,
            domain=request.domain,
            prediction=scoreline[0] if scoreline else agg.prediction,
            score=agg.score,
            confidence=agg.confidence,
            risk_level=agg.risk_level,
            explanation=explanation,
            contributors=agg.contributors,
            weights_used=agg.weights_used,
            score_detail=scoreline[1] if scoreline else None,
            tenant_id=tenant_id,
        )
        self.predictions.add(prediction)

        for result in results:
            self.agent_results.add(
                AgentResultModel(
                    prediction_id=prediction.id,
                    agent_name=result.agent,
                    score=result.score,
                    confidence=result.confidence,
                    weight=agg.weights_used.get(result.agent),
                    reasoning=result.reasoning,
                    extra=result.extra,
                    tenant_id=tenant_id,
                )
            )

        logger.info(
            "prediction.created",
            entity=request.entity,
            prediction=agg.prediction,
            confidence=agg.confidence,
            conflict=agg.conflict,
        )
        record_prediction(request.domain)  # Sprint 14 metric
        return prediction

    def get(self, prediction_id: uuid.UUID) -> PredictionModel | None:
        return self.predictions.get_with_agents(prediction_id)

    def list_predictions(
        self, *, limit: int = 50, offset: int = 0, tenant_id: str | None = None
    ) -> tuple[list[PredictionModel], int]:
        return self.predictions.list_with_total(limit=limit, offset=offset, tenant_id=tenant_id)

    def _worldcup_finished_records(self) -> list[dict]:
        """One record per finished World Cup match.

        Per match, take the prediction from the last snapshot *before kickoff*
        (so prior-matchday form is used but the match's own result never leaks)
        and the actual score from any post-match snapshot.
        """
        rows = list(
            self.session.scalars(
                select(PredictionModel)
                .where(PredictionModel.domain == "worldcup")
                .order_by(PredictionModel.created_at)
            )
        )
        by_entity: dict[str, list[PredictionModel]] = defaultdict(list)
        for row in rows:
            by_entity[row.entity].append(row)

        records: list[dict] = []
        for snapshots in by_entity.values():
            actual = next(
                (
                    (s.score_detail or {}).get("actual")
                    for s in snapshots
                    if (s.score_detail or {}).get("actual")
                ),
                None,
            )
            if not actual:
                continue
            kickoff = _parse_kickoff((snapshots[0].score_detail or {}).get("utc_date"))
            chosen = snapshots[0]
            if kickoff is not None:
                before = [s for s in snapshots if _as_utc(s.created_at) < kickoff]
                if before:
                    chosen = before[-1]
            detail = chosen.score_detail or {}
            records.append(
                {
                    "entity": chosen.entity,
                    "predicted": detail.get("predicted"),
                    "p_home_win": detail.get("p_home_win", 0.0),
                    "p_draw": detail.get("p_draw", 0.0),
                    "p_away_win": detail.get("p_away_win", 0.0),
                    "home_strength": detail.get("home_strength"),
                    "away_strength": detail.get("away_strength"),
                    "actual": actual,
                }
            )
        return records

    def worldcup_accuracy(self) -> AccuracyReport:
        """Score persisted World Cup snapshots against real results."""
        return evaluate(self._worldcup_finished_records())

    def calibrate_score_model(self, *, apply: bool = True) -> Calibration | None:
        """Fit the score-model constants to results so far; apply them if asked.

        Dormant until enough matches are played (returns None), so the seed values
        stand at the start of the tournament. Feeds [[Sprint WC-6]].
        """
        samples: list[dict] = []
        for record in self._worldcup_finished_records():
            if record["home_strength"] is None or record["away_strength"] is None:
                continue
            try:
                home_goals, away_goals = (int(x) for x in record["actual"].split("-"))
            except (ValueError, AttributeError):
                continue
            samples.append(
                {
                    "home_strength": record["home_strength"],
                    "away_strength": record["away_strength"],
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                }
            )
        result = calibrate(samples)
        if result is not None and apply:
            set_params(result.base_goals, result.strength_sensitivity)
            logger.info(
                "score.calibrated",
                base_goals=result.base_goals,
                strength_sensitivity=result.strength_sensitivity,
                samples=result.samples,
            )
        return result

    # ------------------------------------------------------------------ #
    # Accuracy tracking, benchmarking & auto-weighting (Sprints 10 & 11)
    # ------------------------------------------------------------------ #
    def record_outcome(
        self,
        prediction_id: uuid.UUID,
        actual: str,
        *,
        actual_score: float | None = None,
        notes: str | None = None,
    ) -> OutcomeModel | None:
        """Attach (or update) the real-world result of a prediction (story 10-1)."""
        prediction = self.predictions.get(prediction_id)
        if prediction is None:
            return None
        correct = prediction.prediction.strip().lower() == actual.strip().lower()
        existing = self.outcomes.for_prediction(prediction_id)
        if existing is not None:
            existing.actual = actual
            existing.actual_score = actual_score
            existing.correct = correct
            existing.notes = notes
            self.session.flush()
            return existing
        outcome = OutcomeModel(
            prediction_id=prediction_id,
            actual=actual,
            actual_score=actual_score,
            correct=correct,
            notes=notes,
            tenant_id=prediction.tenant_id,
        )
        return self.outcomes.add(outcome)

    def _agent_observations(self) -> list[dict]:
        """One observation per agent per resolved prediction (label from outcome)."""
        observations: list[dict] = []
        for outcome in self.outcomes.all_with_predictions():
            label = realised_label(outcome.actual)
            prediction = outcome.prediction
            if prediction is None:
                continue
            for ar in prediction.agent_results:
                observations.append(
                    {
                        "agent": ar.agent_name,
                        "score": ar.score,
                        "confidence": ar.confidence,
                        "weight": ar.weight,
                        "label": label,
                    }
                )
        return observations

    def agent_accuracy(self) -> dict[str, float]:
        """Directional hit-rate per agent over all resolved predictions."""
        return agent_accuracy(self._agent_observations())

    def benchmark(self) -> tuple[int, list[AgentBenchmark]]:
        """Per-agent benchmark report + the count of usable (labelled) predictions."""
        observations = self._agent_observations()
        evaluated = sum(1 for o in observations if o["label"] is not None) // max(
            1, len({o["agent"] for o in observations})
        )
        return evaluated, benchmark_agents(observations)

    def _apply_accuracy_weighting(
        self, accuracy: dict[str, float], samples: int, *, source: str
    ) -> tuple[bool, int, dict[str, float], dict[str, float]]:
        """Shared loop: nudge weights from accuracy + persist a snapshot.

        Used by both the outcome-driven (Sprint 10) and feedback-driven
        (Sprint 12) closed loops. Guardrails (clamp + smoothing) live in the
        WeightManager, so a single noisy window can't collapse the vector.
        """
        if not accuracy:
            return False, 0, {}, self.weights.current()

        before = self.weights.current()
        weights = self.weights.adjust_from_accuracy(accuracy)
        adjusted = weights != before

        per_agent = samples // max(1, len(accuracy))
        for agent, acc in accuracy.items():
            self.session.add(
                AccuracySnapshotModel(
                    agent_name=agent,
                    accuracy=acc,
                    sample_size=per_agent,
                    weight_after=weights.get(agent),
                )
            )
        overall = round(sum(accuracy.values()) / len(accuracy), 3)
        self.session.add(
            AccuracySnapshotModel(agent_name="__global__", accuracy=overall, sample_size=per_agent)
        )
        self.session.flush()
        logger.info("weights.adjusted", source=source, samples=samples, adjusted=adjusted)
        return adjusted, samples, accuracy, weights

    def recalculate_weights(self) -> tuple[bool, int, dict[str, float], dict[str, float]]:
        """Recompute agent accuracy from real outcomes → nudge weights (story 10-3/10-4)."""
        observations = self._agent_observations()
        accuracy = agent_accuracy(observations)
        samples = sum(1 for o in observations if o["label"] is not None)
        return self._apply_accuracy_weighting(accuracy, samples, source="outcomes")

    def ingest_worldcup_outcomes(self) -> int:
        """Auto-attach real World Cup results to full-agent predictions (no human).

        Targets persisted ``worldcup`` predictions that carry per-agent results
        (i.e. were produced by the full pipeline) and have no outcome yet, then
        records the real scoreline football-data.org reports. This is what makes
        the feedback loop autonomous: the engine learns from finished matches.
        """
        from app.connectors.worldcup import WorldCupConnector

        results = WorldCupConnector().finished_results()
        if not results:
            return 0
        rows = self.session.scalars(
            select(PredictionModel).where(PredictionModel.domain == "worldcup")
        )
        ingested = 0
        for pred in rows:
            if pred.outcome is not None or not pred.agent_results or " vs " not in pred.entity:
                continue
            home, away = (part.strip() for part in pred.entity.split(" vs ", 1))
            actual = results.get((home, away))
            if not actual:
                continue
            self.record_outcome(pred.id, actual)
            ingested += 1
        if ingested:
            logger.info("worldcup.outcomes.ingested", count=ingested)
        return ingested

    def autonomous_learn(self) -> dict:
        """Closed, human-free loop: ingest real WC results → re-tune agent weights.

        Replaces the human verdict of Sprint 12 with the tournament's own ground
        truth. Same guardrailed weighting (clamp + smoothing), so a noisy window
        can't destabilise the vector.
        """
        ingested = self.ingest_worldcup_outcomes()
        adjusted, samples, accuracy, weights = self.recalculate_weights()
        summary = {
            "ingested": ingested,
            "adjusted": adjusted,
            "samples": samples,
            "accuracy": accuracy,
        }
        logger.info("autonomous.learn", ingested=ingested, samples=samples, adjusted=adjusted)
        return summary

    # ------------------------------------------------------------------ #
    # Human-in-the-loop feedback & closed-loop self-improvement (Sprint 12)
    # ------------------------------------------------------------------ #
    def record_feedback(
        self,
        prediction_id: uuid.UUID,
        verdict: str,
        *,
        validator: str | None = None,
        corrected_prediction: str | None = None,
        comment: str | None = None,
    ) -> FeedbackModel | None:
        """Store a human validation/correction + its reward signal (story 12-1)."""
        prediction = self.predictions.get(prediction_id)
        if prediction is None:
            return None
        feedback = FeedbackModel(
            prediction_id=prediction_id,
            validator=validator,
            verdict=verdict,
            corrected_prediction=corrected_prediction,
            reward=feedback_reward(verdict, corrected_prediction),
            comment=comment,
            tenant_id=prediction.tenant_id,
        )
        return self.feedback.add(feedback)

    def _feedback_observations(self) -> list[dict]:
        """One observation per agent per human-reviewed prediction (label from verdict)."""
        observations: list[dict] = []
        for fb in self.feedback.all():
            prediction = self.predictions.get(fb.prediction_id)
            if prediction is None:
                continue
            label = label_from_feedback(
                fb.verdict, prediction.prediction, fb.corrected_prediction
            )
            for ar in prediction.agent_results:
                observations.append(
                    {
                        "agent": ar.agent_name,
                        "score": ar.score,
                        "confidence": ar.confidence,
                        "weight": ar.weight,
                        "label": label,
                    }
                )
        return observations

    def learn_from_feedback(self) -> tuple[bool, int, dict[str, float], dict[str, float]]:
        """Close the loop: human verdicts reshape agent weights (story 12-2/12-3).

        Demonstrates the Sprint 12 DoD — a correction changes future behaviour —
        through the same guardrailed weighting used for real outcomes.
        """
        observations = self._feedback_observations()
        accuracy = agent_accuracy(observations)
        samples = sum(1 for o in observations if o["label"] is not None)
        return self._apply_accuracy_weighting(accuracy, samples, source="feedback")

    def rollback_weights(self) -> dict[str, float]:
        """Guardrail: drop learned weights back to the configured defaults."""
        weights = self.weights.reset()
        logger.info("weights.rolledback", weights=weights)
        return weights

    @staticmethod
    def predict_group(group: str) -> GroupPrediction | None:
        """Predict a World Cup group's standings + qualifiers (Monte-Carlo)."""
        from app.connectors.worldcup import WorldCupConnector

        data = WorldCupConnector().group_data(group)
        if data is None:
            return None
        label, teams, fixtures, results = data
        return simulate_group(label, teams, fixtures, results=results)

    @staticmethod
    def predict_tournament() -> TournamentPrediction | None:
        """Predict the whole bracket + champion (Monte-Carlo over the 12 groups)."""
        from app.connectors.worldcup import WorldCupConnector

        all_groups = WorldCupConnector().all_group_data()
        if not all_groups or len(all_groups) != 12:  # need the full 12-group field
            return None
        return simulate_tournament(all_groups)

    @staticmethod
    def predict_matchday(matchday: int) -> list[dict] | None:
        """Predict every World Cup match of a given group-stage matchday."""
        from app.connectors.worldcup import WorldCupConnector, team_strength

        matches = WorldCupConnector().matchday_fixtures(matchday)
        if matches is None:
            return None
        out: list[dict] = []
        for match in matches:
            home = (match.get("homeTeam") or {}).get("name")
            away = (match.get("awayTeam") or {}).get("name")
            if not home or not away:
                continue
            home_strength = team_strength(home)
            away_strength = team_strength(away)
            sp = predict_scoreline(home_strength, away_strength, neutral=True)
            actual = None
            if match.get("status") == "FINISHED":
                full_time = (match.get("score") or {}).get("fullTime") or {}
                if full_time.get("home") is not None and full_time.get("away") is not None:
                    actual = f"{full_time['home']}-{full_time['away']}"
            out.append(
                {
                    "home": home,
                    "away": away,
                    "group": match.get("group"),
                    "matchday": match.get("matchday"),
                    "utc_date": match.get("utcDate"),
                    "status": match.get("status"),
                    "predicted": sp.scoreline,
                    "p_home_win": sp.p_home_win,
                    "p_draw": sp.p_draw,
                    "p_away_win": sp.p_away_win,
                    "home_strength": home_strength,
                    "away_strength": away_strength,
                    "actual": actual,
                }
            )
        out.sort(key=lambda x: x.get("utc_date") or "")
        return out

    @staticmethod
    def _scoreline_for(request: PredictionRequest) -> tuple[str, dict] | None:
        """Predict the final score for a resolvable two-team matchup (else None)."""
        for connector in PredictionService._connectors_for(request.domain):
            matchup_fn = getattr(connector, "matchup_metrics", None)
            if not callable(matchup_fn):
                continue
            matchup = matchup_fn(request.entity)
            if matchup is None or matchup.away is None:
                continue
            # World Cup ties are at neutral venues -> no first-team home bias, so
            # the favourite advances regardless of the order the teams are named.
            neutral = getattr(connector, "domain", None) == "worldcup"
            prediction = predict_scoreline(
                matchup.home.strength,
                matchup.away.strength,
                neutral=neutral,
                knockout=request.knockout,
            )
            detail = asdict(prediction)
            detail["home_team"] = matchup.home.team
            detail["away_team"] = matchup.away.team
            winner_team = None
            if prediction.winner is not None:
                winner_team = (
                    matchup.home.team if prediction.winner == "home" else matchup.away.team
                )
            detail["winner_team"] = winner_team
            label = (
                f"{matchup.home.team} {prediction.home_goals}-"
                f"{prediction.away_goals} {matchup.away.team}"
            )
            # On a knockout draw, the scoreline alone hides the winner -> append it.
            if prediction.decided_by == "shootout" and winner_team is not None:
                label += f" → {winner_team} (ET/pens)"
            return label, detail
        return None

    @staticmethod
    def _connectors_for(domain: str | None) -> list[object]:
        """Pick data connectors to wire in for a domain (real-data sources)."""
        normalised = (domain or "").lower()
        if normalised in ("worldcup", "world cup", "coupe du monde", "mondial", "wc"):
            from app.connectors.worldcup import WorldCupConnector

            return [WorldCupConnector()]
        if normalised in ("sports", "football", "soccer"):
            from app.connectors.openligadb import OpenLigaDBConnector

            return [OpenLigaDBConnector()]
        return []

    @staticmethod
    def _run_agents(request: PredictionRequest) -> list[AgentResult]:
        graph = build_graph()
        context = dict(request.context)
        connectors = PredictionService._connectors_for(request.domain)
        if connectors:
            context["_connectors"] = [*context.get("_connectors", []), *connectors]
        # Sprint 07: expose the MCP manager so the Contextual agent can consume
        # an MCP resource. Sources self-scope (the demo abstains on World Cup).
        if settings.mcp_enabled and "_mcp" not in context:
            from app.mcp import get_mcp_manager

            context["_mcp"] = get_mcp_manager()
        state = {
            "entity": request.entity,
            "domain": request.domain,
            "context": context,
        }
        final = graph.invoke(state)
        results = list(final.get("results", []))
        if not results:
            raise RuntimeError("Orchestration produced no agent results")
        return results


def get_prediction_service(session: Session) -> PredictionService:
    """Factory for the API layer — wire with ``Depends(get_session)`` (Sprint 06)."""
    return PredictionService(session)
