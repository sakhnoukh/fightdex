"""Tests for regression and classification evaluation metrics.

The assignment requires metrics beyond ranking (hit@k, NDCG):
- Classification framing: precision, recall, F1 treating recommendation as binary classification
- Regression framing: RMSE and MAE on predicted relevance scores vs ground truth

These tests define the expected API before the functions are implemented.
"""
from __future__ import annotations

import math
import pytest


# ---------------------------------------------------------------------------
# Helpers — minimal ground truth fixtures
# ---------------------------------------------------------------------------

# y_true: one relevant item per query (same shape as existing leave-one-out protocol)
# y_pred: ranked list of recommended items
PERFECT_TRUE  = ["A", "B", "C"]
PERFECT_PRED  = [["A", "X", "Y"], ["B", "X", "Y"], ["C", "X", "Y"]]  # hit at rank 1 each

PARTIAL_TRUE  = ["A", "B", "C"]
PARTIAL_PRED  = [["A", "X", "Y"], ["X", "Y", "Z"], ["X", "Y", "Z"]]  # 1/3 hit

NO_HIT_TRUE   = ["A", "B", "C"]
NO_HIT_PRED   = [["X", "Y", "Z"], ["X", "Y", "Z"], ["X", "Y", "Z"]]  # 0 hits

K = 3


# ===========================================================================
# Classification metrics: binary_precision_recall_f1
# ===========================================================================

def test_classification_perfect_precision(data=None):
    from pokecoach.evaluation import binary_classification_metrics
    result = binary_classification_metrics(PERFECT_TRUE, PERFECT_PRED, k=K)
    assert result["precision"] == pytest.approx(1.0), "Perfect hit → precision=1.0"


def test_classification_perfect_recall(data=None):
    from pokecoach.evaluation import binary_classification_metrics
    result = binary_classification_metrics(PERFECT_TRUE, PERFECT_PRED, k=K)
    assert result["recall"] == pytest.approx(1.0), "All items retrieved → recall=1.0"


def test_classification_perfect_f1():
    from pokecoach.evaluation import binary_classification_metrics
    result = binary_classification_metrics(PERFECT_TRUE, PERFECT_PRED, k=K)
    assert result["f1"] == pytest.approx(1.0), "Perfect precision + recall → F1=1.0"


def test_classification_no_hit_precision():
    from pokecoach.evaluation import binary_classification_metrics
    result = binary_classification_metrics(NO_HIT_TRUE, NO_HIT_PRED, k=K)
    assert result["precision"] == pytest.approx(0.0)


def test_classification_no_hit_recall():
    from pokecoach.evaluation import binary_classification_metrics
    result = binary_classification_metrics(NO_HIT_TRUE, NO_HIT_PRED, k=K)
    assert result["recall"] == pytest.approx(0.0)


def test_classification_no_hit_f1():
    from pokecoach.evaluation import binary_classification_metrics
    result = binary_classification_metrics(NO_HIT_TRUE, NO_HIT_PRED, k=K)
    assert result["f1"] == pytest.approx(0.0)


def test_classification_partial_hit():
    """1/3 queries answered correctly → precision = recall = F1 ≈ 0.333."""
    from pokecoach.evaluation import binary_classification_metrics
    result = binary_classification_metrics(PARTIAL_TRUE, PARTIAL_PRED, k=K)
    assert result["precision"] == pytest.approx(1 / 3, abs=1e-6)
    assert result["recall"] == pytest.approx(1 / 3, abs=1e-6)


def test_classification_returns_expected_keys():
    from pokecoach.evaluation import binary_classification_metrics
    result = binary_classification_metrics(PERFECT_TRUE, PERFECT_PRED, k=K)
    assert "precision" in result
    assert "recall" in result
    assert "f1" in result


def test_classification_values_in_unit_interval():
    from pokecoach.evaluation import binary_classification_metrics
    result = binary_classification_metrics(PARTIAL_TRUE, PARTIAL_PRED, k=K)
    for key, val in result.items():
        assert 0.0 <= val <= 1.0, f"{key}={val} is outside [0, 1]"


def test_classification_f1_harmonic_mean_of_p_and_r():
    """F1 should equal 2*P*R / (P+R)."""
    from pokecoach.evaluation import binary_classification_metrics
    result = binary_classification_metrics(PARTIAL_TRUE, PARTIAL_PRED, k=K)
    p, r = result["precision"], result["recall"]
    expected_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    assert result["f1"] == pytest.approx(expected_f1, abs=1e-6)


def test_classification_k_cutoff_respected():
    """Items beyond position k should not count as hits."""
    from pokecoach.evaluation import binary_classification_metrics
    # true item is at rank 4 (index 3), outside k=3
    y_true = ["D"]
    y_pred = [["A", "B", "C", "D"]]
    result_k3 = binary_classification_metrics(y_true, y_pred, k=3)
    result_k4 = binary_classification_metrics(y_true, y_pred, k=4)
    assert result_k3["precision"] == pytest.approx(0.0)
    assert result_k4["precision"] == pytest.approx(1.0)


# ===========================================================================
# Regression metrics: rmse_at_k and mae_at_k
# ===========================================================================
#
# These treat relevance as a continuous score:
#   - predicted_scores[i] = the model's score for the top-1 recommendation
#   - true_scores[i]       = 1.0 if it's the relevant item, else 0.0
# This framing converts hit@1 into an RMSE/MAE problem.

def test_rmse_perfect_predictions():
    """Predicted scores exactly matching ground truth → RMSE = 0."""
    from pokecoach.evaluation import rmse_score, mae_score
    y_true_scores = [1.0, 0.0, 1.0]
    y_pred_scores = [1.0, 0.0, 1.0]
    assert rmse_score(y_true_scores, y_pred_scores) == pytest.approx(0.0)
    assert mae_score(y_true_scores, y_pred_scores) == pytest.approx(0.0)


def test_rmse_all_wrong():
    """Predicted 1 when true is 0 and vice versa → RMSE = 1.0."""
    from pokecoach.evaluation import rmse_score
    y_true = [1.0, 0.0, 1.0, 0.0]
    y_pred = [0.0, 1.0, 0.0, 1.0]
    assert rmse_score(y_true, y_pred) == pytest.approx(1.0)


def test_mae_all_wrong():
    from pokecoach.evaluation import mae_score
    y_true = [1.0, 0.0, 1.0, 0.0]
    y_pred = [0.0, 1.0, 0.0, 1.0]
    assert mae_score(y_true, y_pred) == pytest.approx(1.0)


def test_rmse_nonnegative():
    from pokecoach.evaluation import rmse_score
    y_true = [1.0, 0.0, 0.5]
    y_pred = [0.8, 0.3, 0.6]
    assert rmse_score(y_true, y_pred) >= 0.0


def test_mae_nonnegative():
    from pokecoach.evaluation import mae_score
    y_true = [1.0, 0.0, 0.5]
    y_pred = [0.8, 0.3, 0.6]
    assert mae_score(y_true, y_pred) >= 0.0


def test_rmse_greater_than_mae_with_outlier():
    """RMSE penalises large errors more than MAE."""
    from pokecoach.evaluation import rmse_score, mae_score
    # One large error dominates RMSE more than MAE
    y_true = [1.0, 0.0, 0.0, 0.0]
    y_pred = [0.0, 0.0, 0.0, 0.0]  # miss one item
    assert rmse_score(y_true, y_pred) >= mae_score(y_true, y_pred)


def test_rmse_known_value():
    """Sanity-check against a manually computed RMSE."""
    from pokecoach.evaluation import rmse_score
    # errors: [0.5, 0.5] → MSE = 0.25 → RMSE = 0.5
    y_true = [1.0, 0.0]
    y_pred = [0.5, 0.5]
    assert rmse_score(y_true, y_pred) == pytest.approx(0.5)


def test_mae_known_value():
    """Sanity-check against a manually computed MAE."""
    from pokecoach.evaluation import mae_score
    y_true = [1.0, 0.0]
    y_pred = [0.5, 0.5]
    assert mae_score(y_true, y_pred) == pytest.approx(0.5)


def test_rmse_empty_raises():
    from pokecoach.evaluation import rmse_score
    with pytest.raises((ValueError, ZeroDivisionError)):
        rmse_score([], [])


def test_mae_empty_raises():
    from pokecoach.evaluation import mae_score
    with pytest.raises((ValueError, ZeroDivisionError)):
        mae_score([], [])


def test_rmse_mismatched_lengths_raises():
    from pokecoach.evaluation import rmse_score
    with pytest.raises((ValueError, AssertionError)):
        rmse_score([1.0, 0.0], [0.5])


# ===========================================================================
# Integration: metrics work on real model output
# ===========================================================================

def test_classification_metrics_on_popularity(tmp_path):
    """binary_classification_metrics should run end-to-end on real model output."""
    from pokecoach.models import load_model_data, PopularityRecommender
    from pokecoach.evaluation import binary_classification_metrics
    import pandas as pd
    from tests.conftest import ARTIFACTS_ROOT

    data = load_model_data(ARTIFACTS_ROOT)
    model = PopularityRecommender(data)
    teams = pd.read_csv(ARTIFACTS_ROOT / "eval" / "reconstruction_teams.csv")

    y_true, y_pred = [], []
    for team in teams.groupby("team_id")["pokemon"].apply(list):
        for idx, hidden in enumerate(team):
            partial = team[:idx] + team[idx + 1:]
            y_true.append(hidden)
            y_pred.append(model.recommend(partial, k=5))
        if len(y_true) >= 30:  # small sample to keep test fast
            break

    result = binary_classification_metrics(y_true, y_pred, k=5)
    assert 0.0 <= result["f1"] <= 1.0


def test_rmse_on_popularity_top1_scores(tmp_path):
    """rmse_score should run end-to-end converting top-1 hit to binary score."""
    from pokecoach.models import load_model_data, PopularityRecommender
    from pokecoach.evaluation import rmse_score
    import pandas as pd
    from tests.conftest import ARTIFACTS_ROOT

    data = load_model_data(ARTIFACTS_ROOT)
    model = PopularityRecommender(data)
    teams = pd.read_csv(ARTIFACTS_ROOT / "eval" / "reconstruction_teams.csv")

    y_true_scores, y_pred_scores = [], []
    for team in teams.groupby("team_id")["pokemon"].apply(list):
        for idx, hidden in enumerate(team):
            partial = team[:idx] + team[idx + 1:]
            recs = model.recommend(partial, k=1)
            y_true_scores.append(1.0)          # the hidden item is always relevant
            y_pred_scores.append(1.0 if hidden in recs else 0.0)
        if len(y_true_scores) >= 30:
            break

    rmse = rmse_score(y_true_scores, y_pred_scores)
    assert rmse >= 0.0
