"""Tests for the Poisson final-score model (World Cup)."""

from __future__ import annotations

from app.prediction.score import (
    ScorePrediction,
    expected_goals,
    predict_scoreline,
    shootout_prob,
)


def test_expected_goals_favour_the_stronger_side() -> None:
    lam_home, lam_away = expected_goals(0.9, 0.3)
    assert lam_home > lam_away


def test_strong_vs_weak_predicts_a_home_win() -> None:
    sp = predict_scoreline(0.95, 0.2)
    assert isinstance(sp, ScorePrediction)
    assert sp.home_goals >= sp.away_goals
    assert sp.p_home_win > 0.6
    assert abs(sp.p_home_win + sp.p_draw + sp.p_away_win - 1.0) < 0.02


def test_top_scorelines_are_ranked() -> None:
    sp = predict_scoreline(0.8, 0.5)
    assert len(sp.top_scorelines) == 5
    probs = [t["p"] for t in sp.top_scorelines]
    assert probs == sorted(probs, reverse=True)
    assert sp.top_scorelines[0]["score"] == sp.scoreline  # mode == most likely


def test_even_match_is_balanced_with_real_draw_chance() -> None:
    sp = predict_scoreline(0.6, 0.6)
    assert abs(sp.p_home_win - sp.p_away_win) < 0.2
    assert sp.p_draw > 0.15


def test_reversing_the_order_flips_the_favourite() -> None:
    strong_home = predict_scoreline(0.9, 0.3)
    weak_home = predict_scoreline(0.3, 0.9)
    assert strong_home.p_home_win > strong_home.p_away_win
    assert weak_home.p_away_win > weak_home.p_home_win


# --------------------------------------------------------------------------- #
# Knockout: a winner is forced (extra time / penalties)
# --------------------------------------------------------------------------- #
def test_shootout_prob_is_a_tilted_coin_flip() -> None:
    assert shootout_prob(0.6, 0.6) == 0.5
    assert 0.5 < shootout_prob(0.9, 0.3) <= 0.75
    assert shootout_prob(0.3, 0.9) == round(1.0 - shootout_prob(0.9, 0.3), 3)


def test_group_stage_allows_a_draw_with_no_winner() -> None:
    sp = predict_scoreline(0.6, 0.6, knockout=False)
    assert sp.knockout is False
    assert sp.winner is None
    assert sp.p_home_advance is None


def test_knockout_forces_a_winner_on_a_likely_draw() -> None:
    sp = predict_scoreline(0.6, 0.6, knockout=True)  # evenly matched -> likely 1-1
    assert sp.knockout is True
    assert sp.winner in {"home", "away"}
    assert sp.p_home_advance is not None and sp.p_away_advance is not None
    assert abs(sp.p_home_advance + sp.p_away_advance - 1.0) < 0.02  # no draw left
    assert sp.decided_by == "shootout"  # regulation most likely level


def test_knockout_stronger_side_advances() -> None:
    sp = predict_scoreline(0.9, 0.3, knockout=True)
    assert sp.winner == "home"
    assert sp.p_home_advance is not None and sp.p_away_advance is not None
    assert sp.p_home_advance > sp.p_away_advance
    assert sp.p_home_advance > 0.7
