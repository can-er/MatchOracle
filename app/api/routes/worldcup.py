"""World Cup endpoints (Sprint WC-3): group-stage prediction."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.base import get_session
from app.orchestration.service import PredictionService, get_prediction_service
from app.prediction.params import current_params
from app.schemas.worldcup import (
    AccuracyResponse,
    CalibrationResponse,
    GroupPredictionResponse,
    MatchdayResponse,
    TournamentResponse,
)

router = APIRouter(tags=["worldcup"])


@router.get(
    "/worldcup/champion",
    response_model=TournamentResponse,
    summary="Predict the bracket + champion (full-tournament Monte-Carlo)",
)
def predict_champion() -> TournamentResponse:
    result = PredictionService.predict_tournament()
    if result is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "World Cup group data unavailable (need all 12 groups; is the key set?).",
        )
    return TournamentResponse(
        champion=result.champion,
        p_champion=result.p_champion,
        outlook=[asdict(o) for o in result.outlook],
    )


@router.get(
    "/worldcup/accuracy",
    response_model=AccuracyResponse,
    summary="Prediction accuracy so far (predicted vs actual results)",
)
def worldcup_accuracy(session: Session = Depends(get_session)) -> AccuracyResponse:
    report = get_prediction_service(session).worldcup_accuracy()
    return AccuracyResponse(**asdict(report))


@router.get(
    "/worldcup/calibration",
    response_model=CalibrationResponse,
    summary="Active score-model constants + a calibration suggestion from results",
)
def worldcup_calibration(session: Session = Depends(get_session)) -> CalibrationResponse:
    base_goals, strength_sensitivity = current_params()
    suggestion = get_prediction_service(session).calibrate_score_model(apply=False)
    return CalibrationResponse(
        base_goals=base_goals,
        strength_sensitivity=strength_sensitivity,
        samples=suggestion.samples if suggestion else 0,
        calibrated=suggestion is not None,
        suggested_base_goals=suggestion.base_goals if suggestion else None,
        suggested_strength_sensitivity=suggestion.strength_sensitivity if suggestion else None,
    )


@router.get(
    "/worldcup/matchday/{matchday}",
    response_model=MatchdayResponse,
    summary="Predict every match of a World Cup group-stage matchday (1, 2 or 3)",
)
def predict_matchday(matchday: int) -> MatchdayResponse:
    matches = PredictionService.predict_matchday(matchday)
    if not matches:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "No matches for that matchday or World Cup data unavailable.",
        )
    return MatchdayResponse(matchday=matchday, matches=matches)


@router.get(
    "/worldcup/groups/{group}",
    response_model=GroupPredictionResponse,
    summary="Predict a World Cup group: standings + qualification probabilities",
)
def predict_group(group: str) -> GroupPredictionResponse:
    result = PredictionService.predict_group(group)
    if result is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "Unknown group or World Cup data unavailable (is the football-data key set?).",
        )
    return GroupPredictionResponse(
        group=result.group,
        standings=[asdict(s) for s in result.standings],
        qualifiers=result.qualifiers,
    )
