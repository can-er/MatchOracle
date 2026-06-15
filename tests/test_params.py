"""Tests for the calibratable score-model params holder (Sprint WC-6)."""

from __future__ import annotations

from app.prediction.params import current_params, set_params
from app.prediction.score import BASE_GOALS, STRENGTH_SENSITIVITY


def test_current_params_defaults_to_the_seed() -> None:
    assert current_params() == (BASE_GOALS, STRENGTH_SENSITIVITY)


def test_set_params_round_trip() -> None:
    set_params(1.55, 2.2)
    assert current_params() == (1.55, 2.2)
