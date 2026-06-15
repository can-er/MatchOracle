"""Tests for the scheduled World Cup refresh task (Sprint WC-5)."""

from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import models  # noqa: F401 — register models
from app.db.base import Base
from app.db.models import Prediction
from app.orchestration.service import PredictionService
from app.tasks.refresh import refresh_worldcup


def _session():
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _match(home: str, away: str, status: str = "TIMED", actual: str | None = None) -> dict:
    return {
        "home": home,
        "away": away,
        "group": "GROUP_A",
        "matchday": 1,
        "utc_date": "2026-06-11T19:00:00Z",
        "status": status,
        "predicted": "2-0",
        "p_home_win": 0.7,
        "p_draw": 0.18,
        "p_away_win": 0.12,
        "actual": actual,
    }


def test_refresh_persists_worldcup_snapshots(monkeypatch) -> None:
    matches = [
        _match("Mexico", "South Africa"),
        _match("France", "Brazil", status="FINISHED", actual="2-1"),
    ]
    monkeypatch.setattr(
        PredictionService, "predict_matchday", lambda matchday: matches if matchday == 1 else None
    )

    session = _session()
    try:
        summary = refresh_worldcup(session)
        assert summary["matches"] == 2
        assert summary["finished"] == 1
        assert summary["matchdays"] == {1: 2}

        rows = list(session.scalars(select(Prediction)))
        assert len(rows) == 2
        assert all(r.domain == "worldcup" for r in rows)
        assert {r.prediction for r in rows} == {"2-0"}
        assert all(r.score_detail is not None for r in rows)
        finished = next(r for r in rows if r.entity.startswith("France"))
        assert finished.score_detail["actual"] == "2-1"
    finally:
        session.close()
