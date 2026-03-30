from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"


@pytest.fixture(scope="session")
def artifacts_root() -> Path:
    return ARTIFACTS_ROOT


@pytest.fixture(scope="session")
def model_data():
    from pokecoach.models import load_model_data
    return load_model_data(ARTIFACTS_ROOT)


@pytest.fixture(scope="session")
def hybrid(model_data):
    from pokecoach.models import HybridRecommender
    return HybridRecommender(model_data)


@pytest.fixture(scope="session")
def popular(model_data):
    from pokecoach.models import PopularityRecommender
    return PopularityRecommender(model_data)
