"""Data-layer tests (Sprint 01): repositories, relationship and cascade."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import models  # noqa: F401 — register models on the metadata
from app.db.base import Base
from app.db.models import AgentResult, Connector, Prediction
from app.repositories.connector_repository import ConnectorRepository
from app.repositories.prediction_repository import AgentResultRepository, PredictionRepository


@pytest.fixture
def session() -> Iterator[Session]:
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    try:
        yield db
    finally:
        db.close()


def _make_prediction(**overrides: Any) -> Prediction:
    fields: dict[str, Any] = {
        "entity": "Acme",
        "prediction": "Positive",
        "score": 0.7,
        "confidence": 0.8,
    }
    fields.update(overrides)
    return Prediction(**fields)


def test_prediction_crud(session: Session) -> None:
    repo = PredictionRepository(session)
    pred = repo.add(_make_prediction())
    session.commit()

    assert repo.get(pred.id) is not None
    assert repo.count() == 1
    items, total = repo.list_with_total()
    assert total == 1
    assert items[0].id == pred.id


def test_agent_results_relationship_and_cascade(session: Session) -> None:
    pred = PredictionRepository(session).add(_make_prediction())
    ar_repo = AgentResultRepository(session)
    for name in ("historical", "trend"):
        ar_repo.add(AgentResult(prediction_id=pred.id, agent_name=name, score=0.6, confidence=0.7))
    session.commit()

    assert len(ar_repo.for_prediction(pred.id)) == 2
    assert {a.agent_name for a in pred.agent_results} == {"historical", "trend"}

    # ORM cascade ("all, delete-orphan") removes children with the parent.
    session.delete(pred)
    session.commit()
    assert ar_repo.for_prediction(pred.id) == []


def test_connector_repository(session: Session) -> None:
    repo = ConnectorRepository(session)
    repo.add(Connector(name="api", type="rest", configuration={}, status="inactive"))
    session.commit()

    assert repo.get_by_name("api") is not None
    assert [c.name for c in repo.list_all()] == ["api"]
