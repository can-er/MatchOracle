"""World Cup prediction refresh (Sprint WC-5).

Run on a schedule during the tournament (the ``scheduler`` service runs this on a
loop). Each pass re-predicts every group-stage match (matchdays 1-3) from the
latest data — picking up real results as matches finish — and persists a
timestamped snapshot, so predictions stay fresh and a history accrues for
accuracy tracking.

Run once manually:  ``python -m app.tasks.refresh``
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import Prediction
from app.logging_config import configure_logging, get_logger
from app.orchestration.service import PredictionService

logger = get_logger(__name__)

GROUP_MATCHDAYS = (1, 2, 3)


def refresh_worldcup(session: Session) -> dict:
    """Re-predict every group-stage match and persist a snapshot. Returns a summary."""
    total = 0
    finished = 0
    per_matchday: dict[int, int] = {}

    for matchday in GROUP_MATCHDAYS:
        matches = PredictionService.predict_matchday(matchday)
        if not matches:
            continue
        per_matchday[matchday] = len(matches)
        for match in matches:
            total += 1
            if match.get("status") == "FINISHED":
                finished += 1
            probs = (match["p_home_win"], match["p_draw"], match["p_away_win"])
            session.add(
                Prediction(
                    entity=f"{match['home']} vs {match['away']}",
                    domain="worldcup",
                    prediction=match["predicted"],
                    score=match["p_home_win"],
                    confidence=max(probs),
                    contributors=[],
                    score_detail=match,
                )
            )

    session.commit()
    service = PredictionService(session)
    report = service.worldcup_accuracy()
    calibration = service.calibrate_score_model()  # dormant until enough matches
    # Autonomous feedback loop (no human): ingest real results for full-agent
    # predictions and re-tune the agent weights from the tournament's own truth.
    learn = service.autonomous_learn()
    session.commit()
    summary = {
        "matchdays": per_matchday,
        "matches": total,
        "finished": finished,
        "evaluated": report.evaluated,
        "outcome_accuracy": report.outcome_accuracy,
        "brier": report.brier,
        "calibrated": calibration is not None,
        "outcomes_ingested": learn["ingested"],
        "weights_adjusted": learn["adjusted"],
        "agent_samples": learn["samples"],
    }
    logger.info("worldcup.refresh", **summary)
    return summary


def main() -> None:
    configure_logging()
    from app.db.base import SessionLocal

    session = SessionLocal()
    try:
        summary = refresh_worldcup(session)
        logger.info("worldcup.refresh.done", **summary)
    except Exception as exc:  # never crash the scheduler loop
        logger.warning("worldcup.refresh.failed", error=str(exc))
    finally:
        session.close()


if __name__ == "__main__":
    main()
