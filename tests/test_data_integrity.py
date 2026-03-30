"""Tests that verify the parsed artifact data is correct after fixing the parsers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"


def test_legal_pool_nonempty():
    legal = pd.read_csv(ARTIFACTS_ROOT / "features" / "legal_pool.csv")
    assert len(legal) >= 10, "Legal pool should have at least 10 pokemon"


def test_legal_pool_has_incineroar():
    legal = pd.read_csv(ARTIFACTS_ROOT / "features" / "legal_pool.csv")
    assert "Incineroar" in legal["pokemon"].values


def test_moveset_no_spreads():
    moveset = pd.read_csv(ARTIFACTS_ROOT / "smogon" / "regg_moveset.csv")
    assert "Spreads" not in moveset["pokemon"].values, (
        "'Spreads' should not appear as a pokemon name — parser fix required"
    )


def test_moveset_incineroar_has_fake_out():
    moveset = pd.read_csv(ARTIFACTS_ROOT / "smogon" / "regg_moveset.csv")
    inc_moves = moveset[moveset["pokemon"] == "Incineroar"]["move"].tolist()
    assert "Fake Out" in inc_moves, f"Incineroar should have Fake Out; got {inc_moves[:5]}"


def test_teammates_valid_pokemon_names():
    teammates = pd.read_csv(ARTIFACTS_ROOT / "smogon" / "regg_teammates.csv")
    legal = pd.read_csv(ARTIFACTS_ROOT / "features" / "legal_pool.csv")
    legal_names = set(legal["pokemon"].tolist())
    pokemon_col = set(teammates["pokemon"].unique())
    # "Moves" appearing as a pokemon name is the known bug
    assert "Moves" not in pokemon_col, (
        f"'Moves' should not be a pokemon name — parser fix required. Got: {pokemon_col}"
    )
    # At least half the pokemon column entries should be real pokemon names
    overlap = pokemon_col & legal_names
    assert len(overlap) >= 5, f"Too few legal pokemon in teammates pokemon column: {overlap}"


def test_cooccurrence_offdiagonal_nonzero():
    cooc = pd.read_csv(ARTIFACTS_ROOT / "features" / "cooccurrence.csv").set_index("pokemon")
    # Incineroar (#1) and Rillaboom (#2) are the top 2 VGC pokemon — they MUST co-occur
    assert cooc.loc["Incineroar", "Rillaboom"] > 0, (
        "Incineroar+Rillaboom cooccurrence is 0 — teammate parsing is broken"
    )
    # At least some off-diagonal values should be non-trivially positive
    diagonal_mask = (cooc.index.values[:, None] == cooc.columns.values[None, :])
    off_diag = cooc.values[~diagonal_mask]
    assert off_diag.max() > 0.05, "Cooccurrence matrix has no meaningful off-diagonal values"


def test_counter_fire_beats_grass():
    counter = pd.read_csv(ARTIFACTS_ROOT / "features" / "counter_matrix.csv").set_index("pokemon")
    # Incineroar (fire/dark) should have high effectiveness against Rillaboom (grass)
    val = counter.loc["Incineroar", "Rillaboom"]
    assert val > 1.0, f"Expected fire > grass effectiveness, got {val}"


def test_pokemon_types_artifact_exists():
    types_path = ARTIFACTS_ROOT / "features" / "pokemon_types.json"
    assert types_path.exists(), "pokemon_types.json artifact should exist after preprocess"


def test_pokemon_types_incineroar_is_fire():
    import json
    types_path = ARTIFACTS_ROOT / "features" / "pokemon_types.json"
    if not types_path.exists():
        pytest.skip("pokemon_types.json not yet generated")
    types = json.loads(types_path.read_text())
    assert "Incineroar" in types
    assert "fire" in types["Incineroar"], f"Incineroar should be fire type, got {types.get('Incineroar')}"
