"""Prediction / agent-result / outcome repositories (Sprints 01 & 10)."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AgentResult, Feedback, Outcome, Prediction
from app.repositories.base import BaseRepository


class PredictionRepository(BaseRepository[Prediction]):
    model = Prediction

    def list_with_total(
        self, *, limit: int = 50, offset: int = 0, tenant_id: str | None = None
    ) -> tuple[list[Prediction], int]:
        stmt = select(Prediction).order_by(Prediction.created_at.desc())
        count_stmt = select(Prediction)
        if tenant_id is not None:
            stmt = stmt.where(Prediction.tenant_id == tenant_id)
            count_stmt = count_stmt.where(Prediction.tenant_id == tenant_id)
        total = len(list(self.session.scalars(count_stmt).all()))
        items = list(self.session.scalars(stmt.limit(limit).offset(offset)).all())
        return items, total

    def get_with_agents(self, prediction_id: uuid.UUID) -> Prediction | None:
        return self.session.get(Prediction, prediction_id)


class AgentResultRepository(BaseRepository[AgentResult]):
    model = AgentResult

    def for_prediction(self, prediction_id: uuid.UUID) -> list[AgentResult]:
        stmt = select(AgentResult).where(AgentResult.prediction_id == prediction_id)
        return list(self.session.scalars(stmt).all())

    def for_agent(self, agent_name: str, *, limit: int = 1000) -> list[AgentResult]:
        stmt = (
            select(AgentResult)
            .where(AgentResult.agent_name == agent_name)
            .order_by(AgentResult.created_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt).all())


class OutcomeRepository(BaseRepository[Outcome]):
    model = Outcome

    def __init__(self, session: Session) -> None:
        super().__init__(session)

    def for_prediction(self, prediction_id: uuid.UUID) -> Outcome | None:
        stmt = select(Outcome).where(Outcome.prediction_id == prediction_id)
        return self.session.scalars(stmt).first()

    def all_with_predictions(self) -> list[Outcome]:
        stmt = select(Outcome)
        return list(self.session.scalars(stmt).all())


class FeedbackRepository(BaseRepository[Feedback]):
    model = Feedback

    def for_prediction(self, prediction_id: uuid.UUID) -> list[Feedback]:
        stmt = select(Feedback).where(Feedback.prediction_id == prediction_id)
        return list(self.session.scalars(stmt).all())

    def all(self) -> list[Feedback]:
        return list(self.session.scalars(select(Feedback)).all())
