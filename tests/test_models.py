"""Tests for recommender model behaviour."""
from __future__ import annotations

import pytest


PARTIAL_TEAM = ["Incineroar", "Rillaboom"]
OPPONENT = ["Kyogre", "Flutter Mane"]


def test_recommend_returns_k_items(hybrid):
    result = hybrid.recommend(PARTIAL_TEAM, k=5)
    assert len(result) == 5


def test_recommend_excludes_partial_team(hybrid):
    result = hybrid.recommend(PARTIAL_TEAM, k=5)
    assert not set(result) & set(PARTIAL_TEAM), "Recommendations must not include already-chosen pokemon"


def test_recommend_empty_partial_team(hybrid):
    result = hybrid.recommend([], k=5)
    assert len(result) == 5


def test_recommend_single_slot(hybrid):
    full_minus_one = ["Incineroar", "Rillaboom", "Urshifu-Rapid-Strike", "Tornadus", "Flutter Mane"]
    result = hybrid.recommend(full_minus_one, k=1)
    assert len(result) == 1
    assert result[0] not in full_minus_one


def test_hybrid_opponent_changes_output(hybrid):
    no_opp = hybrid.recommend(PARTIAL_TEAM, opponent_context=None, k=10)
    with_opp = hybrid.recommend(PARTIAL_TEAM, opponent_context=OPPONENT, k=10)
    # Counter-weighting with opponent context should produce a different ranking
    assert no_opp != with_opp, "Opponent context should change recommendation order"


def test_popularity_top_is_high_usage(popular, model_data):
    result = popular.recommend([], k=3)
    legal = model_data.legal_pool.set_index("pokemon")
    # All top results should have above-average usage
    avg_usage = legal["usage_pct"].mean()
    for mon in result:
        assert legal.loc[mon, "usage_pct"] > avg_usage, f"{mon} has below-average usage but was top recommended"


def test_recommend_no_duplicates(hybrid):
    result = hybrid.recommend(PARTIAL_TEAM, k=5)
    assert len(result) == len(set(result)), "Recommendations should have no duplicates"


def test_recommend_moveset_returns_moves(model_data):
    from pokecoach.models import recommend_moveset
    moves = recommend_moveset("Incineroar", model_data.moveset, k=4)
    assert len(moves) > 0, "Should return moves for Incineroar"
    assert "Fake Out" in moves, f"Incineroar's top moves should include Fake Out, got {moves}"


def test_recommend_moveset_unknown_pokemon(model_data):
    from pokecoach.models import recommend_moveset
    moves = recommend_moveset("NotAPokemon", model_data.moveset, k=4)
    assert moves == [], "Unknown pokemon should return empty moveset"


def test_type_filter_restricts_types(hybrid, model_data):
    if not model_data.pokemon_types:
        pytest.skip("pokemon_types not loaded — run preprocess first")
    water_mons = [p for p, t in model_data.pokemon_types.items() if "water" in t]
    if not water_mons:
        pytest.skip("No water-type pokemon in legal pool")
    result = hybrid.recommend(["Incineroar"], k=5, type_filter=["water"])
    for mon in result:
        types = model_data.pokemon_types.get(mon, [])
        assert "water" in types, f"{mon} is not water type but was returned with type_filter=['water']"


def test_weight_overrides_change_output(hybrid):
    default = hybrid.recommend(PARTIAL_TEAM, k=10)
    # Heavy viability bias should prefer popular pokemon
    viability_heavy = hybrid.recommend(
        PARTIAL_TEAM, k=10, weight_overrides={"synergy": 0.0, "counter": 0.0, "viability": 1.0, "content": 0.0}
    )
    assert default != viability_heavy, "Weight overrides should change recommendation order"
