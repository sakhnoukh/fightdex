"""Tests for the hyperparameter tuning module."""
from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import ARTIFACTS_ROOT


@pytest.fixture(scope="module")
def tuning_fixtures():
    from pokecoach.models import load_model_data

    data = load_model_data(ARTIFACTS_ROOT)
    teams = pd.read_csv(ARTIFACTS_ROOT / "eval" / "reconstruction_teams.csv")
    counter = pd.read_csv(ARTIFACTS_ROOT / "features" / "counter_matrix.csv").set_index("pokemon")
    return data, teams, counter


def test_tune_hybrid_weights_returns_dataframe(tuning_fixtures):
    from pokecoach.tuning import tune_hybrid_weights

    data, teams, counter = tuning_fixtures
    # Use a tiny grid to keep the test fast
    small_grid = [
        {"synergy": 0.45, "counter": 0.30, "viability": 0.20, "content": 0.05},
        {"synergy": 0.20, "counter": 0.10, "viability": 0.60, "content": 0.10},
    ]
    result = tune_hybrid_weights(data, teams, counter, grid=small_grid, verbose=False)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "hit_rate_5" in result.columns
    assert "synergy" in result.columns


def test_tune_hybrid_weights_sorted_descending(tuning_fixtures):
    from pokecoach.tuning import tune_hybrid_weights

    data, teams, counter = tuning_fixtures
    small_grid = [
        {"synergy": 0.45, "counter": 0.30, "viability": 0.20, "content": 0.05},
        {"synergy": 0.65, "counter": 0.00, "viability": 0.30, "content": 0.05},
        {"synergy": 0.20, "counter": 0.45, "viability": 0.10, "content": 0.25},
    ]
    result = tune_hybrid_weights(data, teams, counter, grid=small_grid, verbose=False)
    scores = result["hit_rate_5"].tolist()
    assert scores == sorted(scores, reverse=True), "Results should be sorted by primary metric descending"


def test_tune_mf_components_svd(tuning_fixtures):
    from pokecoach.tuning import tune_mf_components

    data, teams, counter = tuning_fixtures
    result = tune_mf_components(data, teams, counter, method="svd", component_grid=[8, 16], verbose=False)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert "n_components" in result.columns
    assert "method" in result.columns
    assert all(result["method"] == "svd")


def test_tune_mf_components_nmf(tuning_fixtures):
    from pokecoach.tuning import tune_mf_components

    data, teams, counter = tuning_fixtures
    result = tune_mf_components(data, teams, counter, method="nmf", component_grid=[8, 16], verbose=False)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert all(result["method"] == "nmf")


def test_evaluate_model_returns_expected_keys(tuning_fixtures):
    from pokecoach.models import PopularityRecommender
    from pokecoach.tuning import _evaluate_model

    data, teams, counter = tuning_fixtures
    model = PopularityRecommender(data)
    metrics = _evaluate_model(model, teams, counter, k=5)
    expected_keys = {"hit_rate_5", "ndcg_5", "precision_3", "coverage", "ild", "personalization"}
    assert set(metrics.keys()) == expected_keys
    for key, val in metrics.items():
        assert 0.0 <= val <= 1.0 or key == "ild", f"{key}={val} out of expected range"
