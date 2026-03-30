"""Tests for simulation utilities."""
from __future__ import annotations


def test_check_showdown_running_returns_bool():
    from pokecoach.simulation import check_showdown_running
    result = check_showdown_running()
    assert isinstance(result, bool)


def test_analytical_win_probability_neutral():
    from pokecoach.api import get_win_probability
    # Same team vs itself → scores cancel out → ~0.5
    team = ["Incineroar", "Rillaboom"]
    result = get_win_probability(team, team)
    assert 0.4 <= result["win_probability"] <= 0.6


def test_analytical_win_probability_fire_vs_grass():
    from pokecoach.api import get_win_probability
    # Fire-heavy team vs grass-heavy: each fire mon scores 2.0 vs grass, grass scores 1.0 vs fire
    # Expected win_prob = 2.0 / (2.0 + 1.0) ≈ 0.667
    fire_team = ["Incineroar", "Chi-Yu"]
    grass_team = ["Rillaboom", "Amoonguss"]
    result = get_win_probability(fire_team, grass_team)
    assert result["win_probability"] > 0.6
