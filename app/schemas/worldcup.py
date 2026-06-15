"""World Cup prediction schemas (Sprint WC-3)."""

from __future__ import annotations

from pydantic import BaseModel


class TeamStandingRead(BaseModel):
    team: str
    exp_points: float
    p_first: float
    p_second: float
    p_third: float
    p_qualify: float


class GroupPredictionResponse(BaseModel):
    group: str
    standings: list[TeamStandingRead]
    qualifiers: list[str]


class MatchPrediction(BaseModel):
    home: str
    away: str
    group: str | None = None
    matchday: int | None = None
    utc_date: str | None = None
    status: str
    predicted: str  # most likely scoreline, e.g. "2-1"
    p_home_win: float
    p_draw: float
    p_away_win: float
    actual: str | None = None  # real score once the match is finished


class MatchdayResponse(BaseModel):
    matchday: int
    matches: list[MatchPrediction]


class AccuracyResponse(BaseModel):
    evaluated: int  # finished matches scored so far
    outcome_correct: int
    outcome_accuracy: float  # share of correct 1X2 results
    exact_correct: int
    exact_accuracy: float  # share of exact scorelines
    mean_goal_error: float  # avg |predicted - actual| goals (both sides)
    brier: float  # 1X2 probabilistic error, lower is better
    details: list[dict] = []


class CalibrationResponse(BaseModel):
    base_goals: float  # active value (calibrated if available, else seed)
    strength_sensitivity: float
    samples: int  # finished matches available to calibrate on
    calibrated: bool  # enough samples to fit?
    suggested_base_goals: float | None = None
    suggested_strength_sensitivity: float | None = None


class TeamOutlookRead(BaseModel):
    team: str
    p_r16: float
    p_qf: float
    p_sf: float
    p_final: float
    p_champion: float


class TournamentResponse(BaseModel):
    champion: str  # most likely winner
    p_champion: float
    outlook: list[TeamOutlookRead]  # all teams, sorted by title probability
