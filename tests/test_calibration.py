"""Tests for the score-model calibration optimizer (Sprint WC-6)."""

from __future__ import annotations

from app.prediction.calibration import MIN_SAMPLES, calibrate


def _sample(sh: float, sa: float, hg: int, ag: int) -> dict:
    return {"home_strength": sh, "away_strength": sa, "home_goals": hg, "away_goals": ag}


def test_calibrate_needs_minimum_samples() -> None:
    assert calibrate([_sample(0.6, 0.6, 2, 1)] * (MIN_SAMPLES - 1)) is None


def test_calibrate_base_goals_tracks_the_scoring_level() -> None:
    # Even matchups, so only the overall goal level varies.
    high = [_sample(0.6, 0.6, 3, 2), _sample(0.6, 0.6, 4, 1)] * 8  # 16 high-scoring
    low = [_sample(0.6, 0.6, 0, 0), _sample(0.6, 0.6, 1, 0)] * 8  # 16 low-scoring

    high_fit, low_fit = calibrate(high), calibrate(low)
    assert high_fit is not None and low_fit is not None
    assert high_fit.samples == 16
    assert high_fit.base_goals > low_fit.base_goals
