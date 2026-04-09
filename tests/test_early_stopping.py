"""Tests for early stopping in MatrixFactorizationRecommender.

Early stopping halts NMF/SVD training when the reconstruction loss stops
improving, preventing overfitting and wasted iterations.
"""
from __future__ import annotations

import pytest

from tests.conftest import ARTIFACTS_ROOT


@pytest.fixture(scope="module")
def data():
    from pokecoach.models import load_model_data
    return load_model_data(ARTIFACTS_ROOT)


# ---------------------------------------------------------------------------
# EarlyStoppingCallback unit tests
# ---------------------------------------------------------------------------

def test_early_stopping_not_triggered_on_improvement():
    """Callback should not stop while loss keeps decreasing."""
    from pokecoach.models import EarlyStoppingCallback
    cb = EarlyStoppingCallback(patience=3, min_delta=0.0)
    losses = [1.0, 0.9, 0.8, 0.7, 0.6]
    for loss in losses:
        cb.update(loss)
    assert not cb.should_stop, "Should not stop when loss is still decreasing"


def test_early_stopping_triggered_after_patience():
    """Callback should stop after `patience` rounds with no improvement."""
    from pokecoach.models import EarlyStoppingCallback
    cb = EarlyStoppingCallback(patience=3, min_delta=0.0)
    # Drop then plateau
    for loss in [1.0, 0.8, 0.8, 0.8, 0.8]:
        cb.update(loss)
    assert cb.should_stop, "Should stop after 3+ rounds with no improvement"


def test_early_stopping_patience_exactly_respected():
    """should_stop should become True at exactly patience rounds without improvement."""
    from pokecoach.models import EarlyStoppingCallback
    cb = EarlyStoppingCallback(patience=2, min_delta=0.0)
    cb.update(1.0)   # improvement
    cb.update(0.8)   # improvement
    cb.update(0.8)   # no improvement, count=1
    assert not cb.should_stop
    cb.update(0.8)   # no improvement, count=2 == patience
    assert cb.should_stop


def test_early_stopping_resets_on_improvement():
    """Counter should reset when a new best is found after a plateau."""
    from pokecoach.models import EarlyStoppingCallback
    cb = EarlyStoppingCallback(patience=2, min_delta=0.0)
    cb.update(1.0)
    cb.update(1.0)  # count=1
    cb.update(0.5)  # improvement — count resets
    assert not cb.should_stop
    assert cb.best_loss < 1.0


def test_early_stopping_min_delta_respected():
    """Improvement smaller than min_delta should not reset the patience counter."""
    from pokecoach.models import EarlyStoppingCallback
    cb = EarlyStoppingCallback(patience=2, min_delta=0.05)
    cb.update(1.0)
    cb.update(0.98)  # delta=0.02 < min_delta → not an improvement
    cb.update(0.96)  # delta=0.02 < min_delta → not an improvement
    assert cb.should_stop, "Tiny improvements below min_delta should not prevent stopping"


def test_early_stopping_tracks_best_loss():
    """best_loss should always reflect the lowest loss seen so far."""
    from pokecoach.models import EarlyStoppingCallback
    cb = EarlyStoppingCallback(patience=5)
    for loss in [1.0, 0.7, 0.9, 0.6, 0.8]:
        cb.update(loss)
    assert cb.best_loss == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# MatrixFactorizationRecommender with early_stopping=True
# ---------------------------------------------------------------------------

def test_mf_early_stopping_flag_accepted(data):
    """MatrixFactorizationRecommender should accept early_stopping parameter."""
    from pokecoach.models import MatrixFactorizationRecommender
    model = MatrixFactorizationRecommender(data, method="nmf", components=8, early_stopping=True)
    assert model is not None


def test_mf_early_stopping_halts_before_max_iter(data):
    """With early stopping enabled, actual iterations should be <= max_iter."""
    from pokecoach.models import MatrixFactorizationRecommender
    model = MatrixFactorizationRecommender(
        data, method="nmf", components=8, early_stopping=True, max_iter=500, patience=5
    )
    # actual_iter is stored on the model after fitting
    assert hasattr(model, "actual_iter"), "Model should expose actual_iter after fitting"
    assert model.actual_iter <= 500


def test_mf_early_stopping_stores_n_iter_no_stopping(data):
    """Without early stopping, actual_iter should equal max_iter (or sklearn's convergence)."""
    from pokecoach.models import MatrixFactorizationRecommender
    model = MatrixFactorizationRecommender(
        data, method="nmf", components=8, early_stopping=False, max_iter=10
    )
    assert hasattr(model, "actual_iter")
    assert model.actual_iter <= 10


def test_mf_early_stopping_still_recommends(data):
    """Model with early stopping should still produce valid recommendations."""
    from pokecoach.models import MatrixFactorizationRecommender
    model = MatrixFactorizationRecommender(data, method="nmf", components=8, early_stopping=True)
    result = model.recommend(["Incineroar", "Rillaboom"], k=5)
    assert len(result) == 5
    assert "Incineroar" not in result
    assert "Rillaboom" not in result


def test_mf_svd_early_stopping_accepted(data):
    """SVD variant should also accept early_stopping without error."""
    from pokecoach.models import MatrixFactorizationRecommender
    model = MatrixFactorizationRecommender(data, method="svd", components=8, early_stopping=True)
    result = model.recommend(["Incineroar"], k=5)
    assert len(result) == 5


def test_mf_early_stopping_patience_param(data):
    """patience parameter should be accepted and influence stopping."""
    from pokecoach.models import MatrixFactorizationRecommender
    # With patience=1 the model should stop very quickly
    fast_stop = MatrixFactorizationRecommender(
        data, method="nmf", components=8, early_stopping=True, patience=1, max_iter=500
    )
    # With patience=50 the model runs longer
    slow_stop = MatrixFactorizationRecommender(
        data, method="nmf", components=8, early_stopping=True, patience=50, max_iter=500
    )
    assert fast_stop.actual_iter <= slow_stop.actual_iter, (
        "Lower patience should result in fewer or equal iterations"
    )
