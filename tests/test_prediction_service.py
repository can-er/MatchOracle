"""End-to-end orchestration smoke test (Sprints 02–05).

Runs the full pipeline against an isolated in-memory SQLite database — no
Postgres/Redis/LLM key required — proving the existing engine actually works:
agents → aggregation → explanation → persistence.
"""

from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import models  # noqa: F401 — register models on the metadata
from app.db.base import Base
from app.db.models import Prediction
from app.orchestration.service import PredictionService
from app.schemas.prediction import PredictionRequest


def _make_session():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_predict_end_to_end_persists_prediction() -> None:
    session = _make_session()
    try:
        service = PredictionService(session)
        prediction = service.predict(PredictionRequest(entity="Test Entity", domain="sports"))
        session.commit()

        assert prediction.id is not None
        assert prediction.prediction in {"Positive", "Negative"}
        assert 0.0 <= prediction.score <= 1.0
        assert 0.0 <= prediction.confidence <= 1.0
        assert prediction.explanation
        # Each registered agent should have produced a persisted result.
        assert len(prediction.agent_results) >= 6
    finally:
        session.close()


def test_predict_worldcup_returns_a_scoreline() -> None:
    session = _make_session()
    try:
        prediction = PredictionService(session).predict(
            PredictionRequest(entity="Argentina vs Saudi Arabia", domain="worldcup")
        )
        session.commit()

        detail = prediction.score_detail
        assert detail is not None
        assert detail["home_team"] == "Argentina"
        assert detail["away_team"] == "Saudi Arabia"
        assert detail["home_goals"] >= detail["away_goals"]  # Argentina is favoured
        assert prediction.prediction == (
            f"Argentina {detail['home_goals']}-{detail['away_goals']} Saudi Arabia"
        )
        total = detail["p_home_win"] + detail["p_draw"] + detail["p_away_win"]
        assert abs(total - 1.0) < 0.02
    finally:
        session.close()


def test_predict_worldcup_knockout_forces_a_winner() -> None:
    session = _make_session()
    try:
        prediction = PredictionService(session).predict(
            PredictionRequest(entity="France vs Brazil", domain="worldcup", knockout=True)
        )
        session.commit()

        detail = prediction.score_detail
        assert detail is not None
        assert detail["knockout"] is True
        assert detail["winner"] in {"home", "away"}
        assert detail["winner_team"] in {"France", "Brazil"}
        # Advancement is forced: no draw outcome remains.
        total = detail["p_home_advance"] + detail["p_away_advance"]
        assert abs(total - 1.0) < 0.02
    finally:
        session.close()


def test_predict_matchday(monkeypatch) -> None:
    fixtures = [
        {
            "matchday": 1,
            "group": "GROUP_A",
            "utcDate": "2026-06-11T19:00:00Z",
            "status": "TIMED",
            "homeTeam": {"name": "Argentina"},
            "awayTeam": {"name": "Saudi Arabia"},
            "score": {"fullTime": {"home": None, "away": None}},
        },
        {
            "matchday": 1,
            "group": "GROUP_B",
            "utcDate": "2026-06-12T19:00:00Z",
            "status": "FINISHED",
            "homeTeam": {"name": "France"},
            "awayTeam": {"name": "Brazil"},
            "score": {"fullTime": {"home": 2, "away": 1}},
        },
    ]
    monkeypatch.setattr(
        "app.connectors.worldcup.WorldCupConnector.matchday_fixtures",
        lambda self, n: fixtures,
    )
    result = PredictionService.predict_matchday(1)
    assert result is not None and len(result) == 2

    arg = next(r for r in result if r["home"] == "Argentina")
    assert "-" in arg["predicted"]
    assert arg["p_home_win"] > arg["p_away_win"]  # Argentina strongly favoured
    france = next(r for r in result if r["home"] == "France")
    assert france["actual"] == "2-1"  # a finished match carries the real score


def test_worldcup_accuracy_from_snapshots() -> None:
    session = _make_session()
    try:
        session.add(
            Prediction(
                entity="France vs Brazil",
                domain="worldcup",
                prediction="2-0",
                score=0.6,
                confidence=0.6,
                contributors=[],
                score_detail={
                    "predicted": "2-0",
                    "p_home_win": 0.6,
                    "p_draw": 0.2,
                    "p_away_win": 0.2,
                    "actual": "2-1",  # France won -> outcome correct, score not exact
                    "utc_date": "2026-06-11T19:00:00Z",
                    "status": "FINISHED",
                },
            )
        )
        session.commit()

        report = PredictionService(session).worldcup_accuracy()
        assert report.evaluated == 1
        assert report.outcome_correct == 1  # 2-0 and 2-1 are both home wins
        assert report.exact_correct == 0
    finally:
        session.close()


def test_calibrate_score_model_applies_when_enough_results() -> None:
    from app.prediction.params import current_params

    session = _make_session()
    try:
        for i in range(14):  # >= MIN_SAMPLES, even strengths, high-scoring
            session.add(
                Prediction(
                    entity=f"Team{i}A vs Team{i}B",
                    domain="worldcup",
                    prediction="3-2",
                    score=0.4,
                    confidence=0.4,
                    contributors=[],
                    score_detail={
                        "predicted": "3-2",
                        "p_home_win": 0.4,
                        "p_draw": 0.3,
                        "p_away_win": 0.3,
                        "home_strength": 0.6,
                        "away_strength": 0.6,
                        "actual": "3-2",
                        "utc_date": "2026-06-11T19:00:00Z",
                        "status": "FINISHED",
                    },
                )
            )
        session.commit()

        result = PredictionService(session).calibrate_score_model()
        assert result is not None
        assert result.samples == 14
        # The fit was applied to the shared params holder.
        assert current_params() == (result.base_goals, result.strength_sensitivity)
    finally:
        session.close()


def test_predict_is_deterministic_for_same_entity() -> None:
    """Heuristic agents are seeded from the entity, so results are reproducible."""
    s1, s2 = _make_session(), _make_session()
    try:
        p1 = PredictionService(s1).predict(PredictionRequest(entity="Same Entity"))
        p2 = PredictionService(s2).predict(PredictionRequest(entity="Same Entity"))
        assert p1.prediction == p2.prediction
        assert p1.score == p2.score
    finally:
        s1.close()
        s2.close()
