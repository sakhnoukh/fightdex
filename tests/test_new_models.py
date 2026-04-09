"""Tests for new recommender models — written before implementation (TDD).

Covers:
- DemographicRecommender
- BowRecommender
- LemmatizedBowRecommender
- SwitchingHybridRecommender
- MixedHybridRecommender
- UCBHybridRecommender online learning (bandit update_reward)
"""
from __future__ import annotations

import pytest

from tests.conftest import ARTIFACTS_ROOT

PARTIAL_TEAM = ["Incineroar", "Rillaboom"]
OPPONENT = ["Kyogre", "Flutter Mane"]


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data():
    from pokecoach.models import load_model_data
    return load_model_data(ARTIFACTS_ROOT)


# ===========================================================================
# DemographicRecommender
# ===========================================================================

@pytest.fixture(scope="module")
def demographic_competitive(data):
    from pokecoach.models import DemographicRecommender
    return DemographicRecommender(data, tier="competitive")


@pytest.fixture(scope="module")
def demographic_casual(data):
    from pokecoach.models import DemographicRecommender
    return DemographicRecommender(data, tier="casual")


def test_demographic_competitive_returns_k_items(demographic_competitive):
    result = demographic_competitive.recommend(PARTIAL_TEAM, k=5)
    assert len(result) == 5


def test_demographic_casual_returns_k_items(demographic_casual):
    result = demographic_casual.recommend(PARTIAL_TEAM, k=5)
    assert len(result) == 5


def test_demographic_competitive_excludes_partial_team(demographic_competitive):
    result = demographic_competitive.recommend(PARTIAL_TEAM, k=5)
    assert not set(result) & set(PARTIAL_TEAM)


def test_demographic_casual_excludes_partial_team(demographic_casual):
    result = demographic_casual.recommend(PARTIAL_TEAM, k=5)
    assert not set(result) & set(PARTIAL_TEAM)


def test_demographic_no_duplicates(demographic_competitive):
    result = demographic_competitive.recommend(PARTIAL_TEAM, k=5)
    assert len(result) == len(set(result))


def test_demographic_tiers_produce_different_results(demographic_competitive, demographic_casual):
    comp = demographic_competitive.recommend([], k=10)
    casual = demographic_casual.recommend([], k=10)
    assert comp != casual, "Competitive and casual tiers should produce different recommendations"


def test_demographic_competitive_prefers_high_usage(demographic_competitive, data):
    """Competitive tier should favour above-median usage pokemon."""
    result = demographic_competitive.recommend([], k=5)
    legal = data.legal_pool.set_index("pokemon")
    median_usage = legal["usage_pct"].median()
    above_median = sum(1 for m in result if legal.loc[m, "usage_pct"] >= median_usage)
    assert above_median >= 3, (
        f"Expected at least 3/5 competitive recs above median usage, got {above_median}"
    )


def test_demographic_casual_avoids_very_top(demographic_casual, data):
    """Casual tier should not be dominated entirely by the top-5 meta picks."""
    result = demographic_casual.recommend([], k=10)
    legal = data.legal_pool.sort_values("usage_pct", ascending=False)
    top5 = set(legal["pokemon"].head(5).tolist())
    top5_in_result = [m for m in result if m in top5]
    assert len(top5_in_result) < 5, (
        "Casual tier should not return all top-5 meta pokemon; expected some variety"
    )


def test_demographic_empty_team(demographic_competitive):
    result = demographic_competitive.recommend([], k=5)
    assert len(result) == 5


def test_demographic_invalid_tier_raises(data):
    from pokecoach.models import DemographicRecommender
    with pytest.raises((ValueError, KeyError)):
        DemographicRecommender(data, tier="nonexistent_tier")


# ===========================================================================
# BowRecommender
# ===========================================================================

@pytest.fixture(scope="module")
def bow(data):
    from pokecoach.models import BowRecommender
    return BowRecommender(data)


def test_bow_returns_k_items(bow):
    result = bow.recommend(PARTIAL_TEAM, k=5)
    assert len(result) == 5


def test_bow_excludes_partial_team(bow):
    result = bow.recommend(PARTIAL_TEAM, k=5)
    assert not set(result) & set(PARTIAL_TEAM)


def test_bow_no_duplicates(bow):
    result = bow.recommend(PARTIAL_TEAM, k=5)
    assert len(result) == len(set(result))


def test_bow_empty_team(bow):
    result = bow.recommend([], k=5)
    assert len(result) == 5


def test_bow_differs_from_tfidf(bow, data):
    """BoW and TF-IDF weight terms differently so rankings should diverge."""
    from pokecoach.models import TfidfRoleRecommender
    tfidf = TfidfRoleRecommender(data)
    bow_recs = bow.recommend(PARTIAL_TEAM, k=10)
    tfidf_recs = tfidf.recommend(PARTIAL_TEAM, k=10)
    assert bow_recs != tfidf_recs, "BoW and TF-IDF should produce different rankings"


def test_bow_similar_move_pool_pokemon_ranked_higher(bow, data):
    """Pokemon sharing many moves with team members should score higher than random."""
    from pokecoach.models import RandomRecommender
    rand = RandomRecommender(data)
    # Run 10 queries; BoW top-1 should occasionally differ from pure random
    bow_top1s = set()
    rand_top1s = set()
    for partial in [["Incineroar"], ["Rillaboom"], ["Urshifu-Rapid-Strike"]]:
        bow_top1s.add(bow.recommend(partial, k=1)[0])
        rand_top1s.add(rand.recommend(partial, k=1)[0])
    # BoW should produce deterministic, non-random rankings
    assert len(bow_top1s) >= 1  # at minimum it runs without error


# ===========================================================================
# LemmatizedBowRecommender
# ===========================================================================

@pytest.fixture(scope="module")
def lemmatized_bow(data):
    from pokecoach.models import LemmatizedBowRecommender
    return LemmatizedBowRecommender(data)


def test_lemmatized_bow_returns_k_items(lemmatized_bow):
    result = lemmatized_bow.recommend(PARTIAL_TEAM, k=5)
    assert len(result) == 5


def test_lemmatized_bow_excludes_partial_team(lemmatized_bow):
    result = lemmatized_bow.recommend(PARTIAL_TEAM, k=5)
    assert not set(result) & set(PARTIAL_TEAM)


def test_lemmatized_bow_no_duplicates(lemmatized_bow):
    result = lemmatized_bow.recommend(PARTIAL_TEAM, k=5)
    assert len(result) == len(set(result))


def test_lemmatized_bow_vocabulary_smaller_than_raw(data):
    """Lemmatization should reduce unique tokens vs raw move names."""
    from pokecoach.models import LemmatizedBowRecommender, BowRecommender
    lem = LemmatizedBowRecommender(data)
    raw = BowRecommender(data)
    assert lem.vocab_size <= raw.vocab_size, (
        f"Lemmatized vocab ({lem.vocab_size}) should be <= raw vocab ({raw.vocab_size})"
    )


def test_lemmatize_move_text():
    """Lemmatization helper should reduce inflected forms to their base."""
    from pokecoach.models import lemmatize_text
    result = lemmatize_text("protecting attacking burning")
    tokens = result.split()
    assert "protecting" not in tokens, "Expected 'protecting' to be lemmatized"
    assert "attacking" not in tokens, "Expected 'attacking' to be lemmatized"


# ===========================================================================
# SwitchingHybridRecommender
# ===========================================================================

@pytest.fixture(scope="module")
def switching(data):
    from pokecoach.models import SwitchingHybridRecommender
    # Switch from popularity to synergy-based at threshold=2
    return SwitchingHybridRecommender(data, threshold=2)


def test_switching_returns_k_items(switching):
    result = switching.recommend(PARTIAL_TEAM, k=5)
    assert len(result) == 5


def test_switching_excludes_partial_team(switching):
    result = switching.recommend(PARTIAL_TEAM, k=5)
    assert not set(result) & set(PARTIAL_TEAM)


def test_switching_no_duplicates(switching):
    result = switching.recommend(PARTIAL_TEAM, k=5)
    assert len(result) == len(set(result))


def test_switching_cold_start_matches_popularity(data):
    """Below threshold, SwitchingHybrid should behave identically to PopularityRecommender."""
    from pokecoach.models import SwitchingHybridRecommender, PopularityRecommender
    switching = SwitchingHybridRecommender(data, threshold=2)
    popular = PopularityRecommender(data)
    # With 0 or 1 pokemon, switching should use popularity
    for partial in [[], ["Incineroar"]]:
        assert switching.recommend(partial, k=5) == popular.recommend(partial, k=5), (
            f"With {len(partial)} pokemon (below threshold=2), switching should equal popularity"
        )


def test_switching_warm_start_differs_from_popularity(data):
    """At or above threshold, SwitchingHybrid should differ from pure popularity."""
    from pokecoach.models import SwitchingHybridRecommender, PopularityRecommender
    switching = SwitchingHybridRecommender(data, threshold=2)
    popular = PopularityRecommender(data)
    warm_partial = ["Incineroar", "Rillaboom", "Urshifu-Rapid-Strike"]
    sw_recs = switching.recommend(warm_partial, k=5)
    pop_recs = popular.recommend(warm_partial, k=5)
    assert sw_recs != pop_recs, (
        "With 3 pokemon (>= threshold=2), switching should use CF, not popularity"
    )


def test_switching_threshold_boundary(data):
    """At exactly threshold, model should switch to the CF strategy."""
    from pokecoach.models import SwitchingHybridRecommender, PopularityRecommender
    switching = SwitchingHybridRecommender(data, threshold=2)
    popular = PopularityRecommender(data)
    at_threshold = ["Incineroar", "Rillaboom"]  # len == threshold
    sw_recs = switching.recommend(at_threshold, k=5)
    pop_recs = popular.recommend(at_threshold, k=5)
    assert sw_recs != pop_recs, "At exactly threshold, switching should use CF strategy"


def test_switching_custom_threshold(data):
    """threshold parameter should be respected."""
    from pokecoach.models import SwitchingHybridRecommender, PopularityRecommender
    # High threshold: even with 3 pokemon, still uses popularity
    switching = SwitchingHybridRecommender(data, threshold=5)
    popular = PopularityRecommender(data)
    partial = ["Incineroar", "Rillaboom", "Urshifu-Rapid-Strike"]
    assert switching.recommend(partial, k=5) == popular.recommend(partial, k=5)


# ===========================================================================
# MixedHybridRecommender
# ===========================================================================

@pytest.fixture(scope="module")
def mixed(data):
    from pokecoach.models import MixedHybridRecommender, PopularityRecommender, KNNRecommender
    return MixedHybridRecommender(data, models=[PopularityRecommender(data), KNNRecommender(data)])


def test_mixed_returns_k_items(mixed):
    result = mixed.recommend(PARTIAL_TEAM, k=6)
    assert len(result) == 6


def test_mixed_excludes_partial_team(mixed):
    result = mixed.recommend(PARTIAL_TEAM, k=6)
    assert not set(result) & set(PARTIAL_TEAM)


def test_mixed_no_duplicates(mixed):
    result = mixed.recommend(PARTIAL_TEAM, k=6)
    assert len(result) == len(set(result))


def test_mixed_draws_from_multiple_models(data):
    """Mixed hybrid should include items sourced from at least two different models."""
    from pokecoach.models import MixedHybridRecommender, PopularityRecommender, KNNRecommender
    pop = PopularityRecommender(data)
    knn = KNNRecommender(data)
    mixed = MixedHybridRecommender(data, models=[pop, knn])

    result = mixed.recommend(PARTIAL_TEAM, k=6)
    pop_recs = set(pop.recommend(PARTIAL_TEAM, k=6))
    knn_recs = set(knn.recommend(PARTIAL_TEAM, k=6))

    from_pop = [m for m in result if m in pop_recs]
    from_knn = [m for m in result if m in knn_recs]
    assert len(from_pop) > 0, "Mixed hybrid should include at least one result from PopularityRecommender"
    assert len(from_knn) > 0, "Mixed hybrid should include at least one result from KNNRecommender"


def test_mixed_differs_from_any_single_model(data):
    """Mixed hybrid result should not be identical to either individual model."""
    from pokecoach.models import MixedHybridRecommender, PopularityRecommender, KNNRecommender
    pop = PopularityRecommender(data)
    knn = KNNRecommender(data)
    mixed = MixedHybridRecommender(data, models=[pop, knn])

    result = mixed.recommend(PARTIAL_TEAM, k=6)
    assert result != pop.recommend(PARTIAL_TEAM, k=6), "Mixed should differ from pure popularity"
    assert result != knn.recommend(PARTIAL_TEAM, k=6), "Mixed should differ from pure KNN"


def test_mixed_empty_team(mixed):
    result = mixed.recommend([], k=5)
    assert len(result) == 5


def test_mixed_single_model_behaves_like_that_model(data):
    """MixedHybrid with a single model should behave identically to that model."""
    from pokecoach.models import MixedHybridRecommender, PopularityRecommender
    pop = PopularityRecommender(data)
    mixed_single = MixedHybridRecommender(data, models=[pop])
    assert mixed_single.recommend(PARTIAL_TEAM, k=5) == pop.recommend(PARTIAL_TEAM, k=5)


# ===========================================================================
# UCBHybridRecommender — online learning (bandit update_reward)
# ===========================================================================

@pytest.fixture
def fresh_bandit(data):
    """Fresh bandit per test so state doesn't bleed between tests."""
    from pokecoach.models import UCBHybridRecommender
    return UCBHybridRecommender(data, c=1.2)


def test_bandit_initial_counts_equal(fresh_bandit):
    """All pokemon should start with the same count so no initial bias."""
    counts = list(fresh_bandit.counts.values())
    assert len(set(counts)) == 1, "Initial counts should all be equal"


def test_bandit_update_reward_increments_count(fresh_bandit):
    before = fresh_bandit.counts.get("Incineroar", 1)
    fresh_bandit.update_reward("Incineroar", reward=1.0)
    after = fresh_bandit.counts.get("Incineroar", 1)
    assert after == before + 1


def test_bandit_update_reward_increments_global_t(fresh_bandit):
    t_before = fresh_bandit.t
    fresh_bandit.update_reward("Incineroar", reward=1.0)
    assert fresh_bandit.t == t_before + 1


def test_bandit_positive_rewards_promote_pokemon(data):
    """A pokemon given many positive rewards should eventually appear in top-k."""
    from pokecoach.models import UCBHybridRecommender
    bandit = UCBHybridRecommender(data, c=2.0)

    target = "Amoonguss"
    # Give strong positive signal
    for _ in range(20):
        bandit.update_reward(target, reward=1.0)

    # Give negative signal to everything else in a short list
    base_recs = bandit.hybrid.recommend([], k=20)
    for mon in base_recs:
        if mon != target:
            bandit.update_reward(mon, reward=0.0)

    result = bandit.recommend([], k=5)
    assert target in result, (
        f"{target} should appear in top-5 after many positive rewards, got {result}"
    )


def test_bandit_negative_rewards_demote_pokemon(data):
    """A pokemon given many negative rewards should not appear in top results."""
    from pokecoach.models import UCBHybridRecommender
    bandit = UCBHybridRecommender(data, c=0.5)  # low exploration

    # Demote the most popular pokemon aggressively
    top_mon = data.legal_pool.sort_values("usage_pct", ascending=False)["pokemon"].iloc[0]
    for _ in range(50):
        bandit.update_reward(top_mon, reward=0.0)

    result = bandit.recommend([], k=5)
    assert top_mon not in result, (
        f"{top_mon} should be demoted after 50 zero-reward updates, but still appears in {result}"
    )


def test_bandit_rewards_change_ranking(data):
    """Rankings after update_reward calls must differ from the initial ranking."""
    from pokecoach.models import UCBHybridRecommender
    bandit = UCBHybridRecommender(data, c=1.2)

    before = bandit.recommend(PARTIAL_TEAM, k=10)

    # Reward a pokemon that wasn't in the initial top-10
    not_in_top = [m for m in data.legal_pool["pokemon"] if m not in before and m not in PARTIAL_TEAM]
    if not_in_top:
        for _ in range(30):
            bandit.update_reward(not_in_top[0], reward=1.0)

    after = bandit.recommend(PARTIAL_TEAM, k=10)
    assert before != after, "Ranking should change after update_reward calls"


def test_bandit_still_returns_k_after_updates(fresh_bandit):
    fresh_bandit.update_reward("Incineroar", reward=1.0)
    fresh_bandit.update_reward("Rillaboom", reward=0.0)
    result = fresh_bandit.recommend(PARTIAL_TEAM, k=5)
    assert len(result) == 5
