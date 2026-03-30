"""Tests for the clean public recommend_team() API."""
from __future__ import annotations

import pytest


def test_recommend_team_fills_slots():
    from pokecoach.api import recommend_team
    result = recommend_team(partial_team=["Incineroar", "Rillaboom", "Urshifu-Rapid-Strike"])
    # 3 chosen → should recommend 3 more to fill a 6-mon team
    assert len(result["recommendations"]) == 3


def test_recommend_team_no_partial():
    from pokecoach.api import recommend_team
    result = recommend_team(partial_team=[])
    assert len(result["recommendations"]) == 6


def test_recommend_team_includes_movesets():
    from pokecoach.api import recommend_team
    result = recommend_team(partial_team=["Incineroar"])
    for rec in result["recommendations"]:
        assert "moves" in rec, f"Recommendation for {rec['pokemon']} missing 'moves' key"


def test_recommend_team_moveset_nonempty():
    from pokecoach.api import recommend_team
    result = recommend_team(partial_team=["Incineroar"])
    # At least some recommended pokemon should have non-empty movesets
    has_moves = [rec for rec in result["recommendations"] if rec["moves"]]
    assert len(has_moves) > 0, "At least some recommendations should have moveset data"


def test_recommend_team_no_duplicates():
    from pokecoach.api import recommend_team
    partial = ["Incineroar", "Rillaboom"]
    result = recommend_team(partial_team=partial)
    rec_names = [r["pokemon"] for r in result["recommendations"]]
    assert not set(rec_names) & set(partial), "Recommendations must not duplicate partial team"
    assert len(rec_names) == len(set(rec_names)), "Recommendations must not have duplicates"


def test_recommend_team_result_structure():
    from pokecoach.api import recommend_team
    result = recommend_team(partial_team=["Incineroar"])
    assert "recommendations" in result
    assert "partial_team" in result
    assert "opponent_context" in result
    for rec in result["recommendations"]:
        assert "pokemon" in rec
        assert "moves" in rec


def test_recommend_team_with_opponent():
    from pokecoach.api import recommend_team
    no_opp = recommend_team(partial_team=["Incineroar"])
    with_opp = recommend_team(partial_team=["Incineroar"], opponent_context=["Kyogre", "Flutter Mane"])
    no_opp_names = [r["pokemon"] for r in no_opp["recommendations"]]
    with_opp_names = [r["pokemon"] for r in with_opp["recommendations"]]
    assert no_opp_names != with_opp_names, "Opponent context should change recommendations"


def test_recommend_team_type_pref(model_data=None):
    from pokecoach.api import recommend_team
    result = recommend_team(partial_team=["Incineroar"], type_preferences=["water"])
    # If water pokemon exist in legal pool, all results should be water type
    recs = result["recommendations"]
    if recs and all(rec.get("types") for rec in recs):
        for rec in recs:
            assert "water" in rec["types"], f"{rec['pokemon']} is not water type"


def test_recommend_team_custom_n():
    from pokecoach.api import recommend_team
    result = recommend_team(partial_team=[], n_recommendations=3)
    assert len(result["recommendations"]) == 3


def test_recommend_team_full_team():
    from pokecoach.api import recommend_team
    full_team = ["Incineroar", "Rillaboom", "Urshifu-Rapid-Strike", "Tornadus", "Flutter Mane", "Amoonguss"]
    result = recommend_team(partial_team=full_team)
    assert len(result["recommendations"]) == 0, "Full team should return empty recommendations"


def test_recommend_team_has_score_breakdown():
    from pokecoach.api import recommend_team
    result = recommend_team(partial_team=["Incineroar"], opponent_context=["Kyogre"])
    for rec in result["recommendations"]:
        assert "scores" in rec, f"{rec['pokemon']} missing 'scores'"
        assert "synergy" in rec["scores"]
        assert "viability" in rec["scores"]
        assert "counter" in rec["scores"]


def test_score_breakdown_values_in_range():
    from pokecoach.api import recommend_team
    result = recommend_team(partial_team=["Incineroar"], opponent_context=["Rillaboom"])
    for rec in result["recommendations"]:
        for key, val in rec["scores"].items():
            assert 0.0 <= val <= 1.0, f"{rec['pokemon']}.scores.{key}={val} out of [0,1]"


def test_get_matchup_matrix_shape():
    from pokecoach.api import get_matchup_matrix
    your_team = ["Incineroar", "Rillaboom"]
    opp_team = ["Kyogre", "Flutter Mane"]
    result = get_matchup_matrix(your_team, opp_team)
    assert result["your_team"] == your_team
    assert result["opp_team"] == opp_team
    assert set(result["matrix"].keys()) == set(your_team)
    for ym in your_team:
        assert set(result["matrix"][ym].keys()) == set(opp_team)


def test_get_matchup_matrix_fire_beats_grass():
    from pokecoach.api import get_matchup_matrix
    result = get_matchup_matrix(["Incineroar"], ["Rillaboom"])
    assert result["matrix"]["Incineroar"]["Rillaboom"] > 1.0


def test_win_probability_range():
    from pokecoach.api import get_win_probability
    result = get_win_probability(["Incineroar", "Rillaboom"], ["Kyogre", "Flutter Mane"])
    assert 0.0 <= result["win_probability"] <= 1.0


def test_win_probability_favours_coverage():
    from pokecoach.api import get_win_probability
    fire_team = ["Incineroar", "Chi-Yu"]
    grass_team = ["Rillaboom", "Amoonguss"]
    result = get_win_probability(fire_team, grass_team)
    assert result["win_probability"] > 0.5
