"""Hyperparameter tuning via grid search over recommender configurations."""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pandas as pd

from pokecoach.config import ProjectConfig
from pokecoach.evaluation import (
    catalog_coverage,
    hit_rate_at_k,
    intra_list_diversity,
    ndcg_at_k,
    personalization,
    precision_at_k,
)
from pokecoach.models import (
    HybridRecommender,
    MatrixFactorizationRecommender,
    ModelData,
    load_model_data,
)
from pokecoach.utils import write_csv, write_json


# ── Grid definitions ──────────────────────────────────────────────────────────
# NOTE: The reconstruction task uses no opponent context, so the counter weight
# (beta) has zero effect on scores.  We keep beta=0.0 in the main synergy /
# viability / content sweep and run a small dedicated counter-weight sweep
# with a fixed synergy+viability baseline so we still report its sensitivity.

_alpha_values = [0.20, 0.35, 0.45, 0.55, 0.65]
_gamma_values = [0.10, 0.20, 0.30, 0.40]
_content_values = [0.05, 0.15, 0.25]

HYBRID_WEIGHT_GRID: list[dict[str, float]] = [
    {"synergy": a, "counter": 0.0, "viability": g, "content": c}
    for a, g, c in itertools.product(_alpha_values, _gamma_values, _content_values)
]

# Small dedicated grid to show counter-weight sensitivity (for the report).
COUNTER_WEIGHT_GRID: list[dict[str, float]] = [
    {"synergy": 0.45, "counter": b, "viability": 0.25, "content": 0.15}
    for b in [0.0, 0.10, 0.20, 0.30, 0.45]
]

SVD_COMPONENT_GRID: list[int] = [8, 16, 24, 32, 48]
NMF_COMPONENT_GRID: list[int] = [8, 16, 24, 32]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _evaluate_model(
    model,
    teams_df: pd.DataFrame,
    counter_matrix: pd.DataFrame,
    k: int = 5,
    max_teams: int | None = None,
) -> dict[str, float]:
    """Run leave-one-out reconstruction and return metric dict.

    Args:
        max_teams: If set, subsample this many teams to speed up the search.
    """
    y_true: list[str] = []
    y_pred: list[list[str]] = []
    grouped = teams_df.groupby("team_id")["pokemon"].apply(list)
    if max_teams is not None and len(grouped) > max_teams:
        grouped = grouped.sample(n=max_teams, random_state=42)
    for team in grouped:
        for idx, hidden in enumerate(team):
            partial = team[:idx] + team[idx + 1:]
            recs = model.recommend(partial, k=k)
            y_true.append(hidden)
            y_pred.append(recs)
    pool = set(teams_df["pokemon"].unique().tolist())
    return {
        "hit_rate_5": hit_rate_at_k(y_true, y_pred, k=5),
        "ndcg_5": ndcg_at_k(y_true, y_pred, k=5),
        "precision_3": precision_at_k(y_true, y_pred, k=3),
        "coverage": catalog_coverage(y_pred, pool),
        "ild": intra_list_diversity(y_pred, counter_matrix),
        "personalization": personalization(y_pred),
    }


# ── Hybrid weight tuning ─────────────────────────────────────────────────────

def tune_hybrid_weights(
    data: ModelData,
    teams_df: pd.DataFrame,
    counter_matrix: pd.DataFrame,
    grid: list[dict[str, float]] | None = None,
    k: int = 5,
    primary_metric: str = "hit_rate_5",
    max_teams: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Grid search over hybrid recommender weight combinations.

    Returns a DataFrame with one row per configuration, sorted by
    *primary_metric* descending.
    """
    if grid is None:
        grid = HYBRID_WEIGHT_GRID

    rows: list[dict[str, Any]] = []
    total = len(grid)

    for i, weights in enumerate(grid):
        hybrid = HybridRecommender(
            data,
            alpha=weights["synergy"],
            beta=weights["counter"],
            gamma=weights["viability"],
        )
        metrics = _evaluate_model(hybrid, teams_df, counter_matrix, k=k, max_teams=max_teams)
        row = {**weights, **metrics}
        rows.append(row)
        if verbose:
            print(
                f"[{i + 1}/{total}] "
                f"syn={weights['synergy']:.2f} ctr={weights['counter']:.2f} "
                f"via={weights['viability']:.2f} cnt={weights['content']:.2f} "
                f"→ {primary_metric}={metrics[primary_metric]:.4f}"
            )

    df = pd.DataFrame(rows).sort_values(primary_metric, ascending=False).reset_index(drop=True)
    return df


# ── MF component tuning ──────────────────────────────────────────────────────

def tune_mf_components(
    data: ModelData,
    teams_df: pd.DataFrame,
    counter_matrix: pd.DataFrame,
    method: str = "svd",
    component_grid: list[int] | None = None,
    k: int = 5,
    primary_metric: str = "hit_rate_5",
    max_teams: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Grid search over the number of latent components for SVD or NMF.
    """
    if component_grid is None:
        component_grid = SVD_COMPONENT_GRID if method == "svd" else NMF_COMPONENT_GRID

    rows: list[dict[str, Any]] = []
    total = len(component_grid)

    for i, n_comp in enumerate(component_grid):
        model = MatrixFactorizationRecommender(data, method=method, components=n_comp)
        metrics = _evaluate_model(model, teams_df, counter_matrix, k=k, max_teams=max_teams)
        row = {"method": method, "n_components": n_comp, **metrics}
        rows.append(row)
        if verbose:
            print(
                f"[{i + 1}/{total}] {method.upper()} components={n_comp} "
                f"→ {primary_metric}={metrics[primary_metric]:.4f}"
            )

    df = pd.DataFrame(rows).sort_values(primary_metric, ascending=False).reset_index(drop=True)
    return df


# ── Full tuning pipeline ─────────────────────────────────────────────────────

def run_tuning(cfg: ProjectConfig, verbose: bool = True) -> dict[str, Path]:
    """
    Run all hyperparameter searches and write results to reports/.

    Outputs:
        reports/tuning_hybrid_weights.csv
        reports/tuning_hybrid_weights.json
        reports/tuning_svd_components.csv
        reports/tuning_svd_components.json
        reports/tuning_nmf_components.csv
        reports/tuning_nmf_components.json
        reports/tuning_best_params.json
    """
    data = load_model_data(cfg.paths["artifacts_root"])
    teams = pd.read_csv(cfg.paths["artifacts_root"] / "eval" / "reconstruction_teams.csv")
    counter = pd.read_csv(
        cfg.paths["artifacts_root"] / "features" / "counter_matrix.csv"
    ).set_index("pokemon")
    reports = cfg.paths["reports_root"]
    outputs: dict[str, Path] = {}

    # 1. Hybrid weight grid search (synergy × viability × content, beta=0)
    if verbose:
        n_grid = len(HYBRID_WEIGHT_GRID)
        print("=" * 60)
        print(f"Hybrid weight grid search ({n_grid} configs)")
        print("=" * 60)
    hybrid_df = tune_hybrid_weights(data, teams, counter, verbose=verbose)
    hybrid_csv = reports / "tuning_hybrid_weights.csv"
    hybrid_json = reports / "tuning_hybrid_weights.json"
    write_csv(hybrid_df, hybrid_csv)
    write_json(hybrid_df.to_dict(orient="records"), hybrid_json)
    outputs["hybrid_weights_csv"] = hybrid_csv
    outputs["hybrid_weights_json"] = hybrid_json

    # 1b. Counter-weight sensitivity sweep (for report completeness)
    if verbose:
        print("=" * 60)
        print(f"Counter-weight sensitivity sweep ({len(COUNTER_WEIGHT_GRID)} configs)")
        print("=" * 60)
    counter_df = tune_hybrid_weights(
        data, teams, counter, grid=COUNTER_WEIGHT_GRID, verbose=verbose,
    )
    counter_csv = reports / "tuning_counter_sensitivity.csv"
    counter_json = reports / "tuning_counter_sensitivity.json"
    write_csv(counter_df, counter_csv)
    write_json(counter_df.to_dict(orient="records"), counter_json)
    outputs["counter_sensitivity_csv"] = counter_csv
    outputs["counter_sensitivity_json"] = counter_json

    # 2. SVD component search
    if verbose:
        print("=" * 60)
        print("SVD latent-component search")
        print("=" * 60)
    svd_df = tune_mf_components(data, teams, counter, method="svd", verbose=verbose)
    svd_csv = reports / "tuning_svd_components.csv"
    svd_json = reports / "tuning_svd_components.json"
    write_csv(svd_df, svd_csv)
    write_json(svd_df.to_dict(orient="records"), svd_json)
    outputs["svd_csv"] = svd_csv
    outputs["svd_json"] = svd_json

    # 3. NMF component search
    if verbose:
        print("=" * 60)
        print("NMF latent-component search")
        print("=" * 60)
    nmf_df = tune_mf_components(data, teams, counter, method="nmf", verbose=verbose)
    nmf_csv = reports / "tuning_nmf_components.csv"
    nmf_json = reports / "tuning_nmf_components.json"
    write_csv(nmf_df, nmf_csv)
    write_json(nmf_df.to_dict(orient="records"), nmf_json)
    outputs["nmf_csv"] = nmf_csv
    outputs["nmf_json"] = nmf_json

    # 4. Summarise best parameters
    best_hybrid = hybrid_df.iloc[0].to_dict()
    best_svd = svd_df.iloc[0].to_dict()
    best_nmf = nmf_df.iloc[0].to_dict()
    best = {
        "hybrid_weights": {
            "synergy": best_hybrid["synergy"],
            "counter": best_hybrid["counter"],
            "viability": best_hybrid["viability"],
            "content": best_hybrid["content"],
            "hit_rate_5": best_hybrid["hit_rate_5"],
            "ndcg_5": best_hybrid["ndcg_5"],
        },
        "svd": {
            "n_components": int(best_svd["n_components"]),
            "hit_rate_5": best_svd["hit_rate_5"],
            "ndcg_5": best_svd["ndcg_5"],
        },
        "nmf": {
            "n_components": int(best_nmf["n_components"]),
            "hit_rate_5": best_nmf["hit_rate_5"],
            "ndcg_5": best_nmf["ndcg_5"],
        },
    }
    best_path = reports / "tuning_best_params.json"
    write_json(best, best_path)
    outputs["best_params"] = best_path

    if verbose:
        print("=" * 60)
        print("Best hybrid weights:")
        print(f"  synergy={best_hybrid['synergy']:.2f}  counter={best_hybrid['counter']:.2f}  "
              f"viability={best_hybrid['viability']:.2f}  content={best_hybrid['content']:.2f}")
        print(f"  hit_rate@5={best_hybrid['hit_rate_5']:.4f}  ndcg@5={best_hybrid['ndcg_5']:.4f}")
        print(f"Best SVD components: {int(best_svd['n_components'])}  "
              f"(hit_rate@5={best_svd['hit_rate_5']:.4f})")
        print(f"Best NMF components: {int(best_nmf['n_components'])}  "
              f"(hit_rate@5={best_nmf['hit_rate_5']:.4f})")
        print("=" * 60)

    return outputs
