from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from pokecoach.config import ProjectConfig
from pokecoach.utils import write_csv, write_json


class Recommender(Protocol):
    def recommend(self, partial_team: list[str], opponent_context: list[str] | None = None, k: int = 5) -> list[str]:
        ...


def hit_rate_at_k(y_true: list[str], y_pred: list[list[str]], k: int = 5) -> float:
    hits = 0
    for truth, pred in zip(y_true, y_pred):
        if truth in pred[:k]:
            hits += 1
    return hits / max(1, len(y_true))


def precision_at_k(y_true: list[str], y_pred: list[list[str]], k: int = 3) -> float:
    total = 0.0
    for truth, pred in zip(y_true, y_pred):
        total += 1.0 / k if truth in pred[:k] else 0.0
    return total / max(1, len(y_true))


def ndcg_at_k(y_true: list[str], y_pred: list[list[str]], k: int = 5) -> float:
    total = 0.0
    for truth, pred in zip(y_true, y_pred):
        try:
            rank = pred[:k].index(truth) + 1
            dcg = 1.0 / np.log2(rank + 1)
        except ValueError:
            dcg = 0.0
        idcg = 1.0
        total += dcg / idcg
    return total / max(1, len(y_true))


def binary_classification_metrics(
    y_true: list[str], y_pred: list[list[str]], k: int = 5
) -> dict[str, float]:
    """Binary classification view of recommendation (1 relevant item per query).

    With exactly one relevant item per query, precision and recall are identical
    (both equal the hit rate), and F1 = that same value.
    """
    hits = sum(1 for t, p in zip(y_true, y_pred) if t in p[:k])
    n = max(1, len(y_true))
    precision = hits / n
    recall = hits / n
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def rmse_score(y_true: list[float], y_pred: list[float]) -> float:
    if len(y_true) == 0:
        raise ValueError("y_true is empty")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    errors = [(t - p) ** 2 for t, p in zip(y_true, y_pred)]
    return float(np.sqrt(np.mean(errors)))


def mae_score(y_true: list[float], y_pred: list[float]) -> float:
    if len(y_true) == 0:
        raise ValueError("y_true is empty")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    return float(np.mean([abs(t - p) for t, p in zip(y_true, y_pred)]))


def catalog_coverage(predictions: list[list[str]], item_pool: set[str]) -> float:
    recommended = {item for recs in predictions for item in recs}
    return len(recommended & item_pool) / max(1, len(item_pool))


def intra_list_diversity(predictions: list[list[str]], counter_matrix: pd.DataFrame) -> float:
    scores: list[float] = []
    for recs in predictions:
        if len(recs) < 2:
            continue
        pair_scores: list[float] = []
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                if recs[i] in counter_matrix.index and recs[j] in counter_matrix.columns:
                    sim = float(counter_matrix.loc[recs[i], recs[j]])
                    pair_scores.append(1.0 / max(sim, 1e-9))
        if pair_scores:
            scores.append(float(np.mean(pair_scores)))
    return float(np.mean(scores)) if scores else 0.0


def personalization(predictions: list[list[str]]) -> float:
    if len(predictions) < 2:
        return 0.0
    jaccards: list[float] = []
    for i in range(len(predictions)):
        for j in range(i + 1, len(predictions)):
            a, b = set(predictions[i]), set(predictions[j])
            jacc = len(a & b) / max(1, len(a | b))
            jaccards.append(jacc)
    return 1.0 - float(np.mean(jaccards))


@dataclass
class EvalResult:
    model_name: str
    hit_rate_5: float
    ndcg_5: float
    precision_3: float
    coverage: float
    ild: float
    personalization: float


def evaluate_reconstruction(
    model: Recommender,
    teams_df: pd.DataFrame,
    counter_matrix: pd.DataFrame,
    model_name: str,
    k: int = 5,
) -> EvalResult:
    y_true: list[str] = []
    y_pred: list[list[str]] = []
    grouped = teams_df.groupby("team_id")["pokemon"].apply(list)
    for team in grouped:
        for idx, hidden in enumerate(team):
            partial = team[:idx] + team[idx + 1 :]
            recs = model.recommend(partial, k=k)
            y_true.append(hidden)
            y_pred.append(recs)
    pool = set(teams_df["pokemon"].unique().tolist())
    return EvalResult(
        model_name=model_name,
        hit_rate_5=hit_rate_at_k(y_true, y_pred, k=5),
        ndcg_5=ndcg_at_k(y_true, y_pred, k=5),
        precision_3=precision_at_k(y_true, y_pred, k=3),
        coverage=catalog_coverage(y_pred, pool),
        ild=intra_list_diversity(y_pred, counter_matrix),
        personalization=personalization(y_pred),
    )


@dataclass
class CVResult:
    model_name: str
    n_folds: int
    hit_rate_5_mean: float
    hit_rate_5_std: float
    ndcg_5_mean: float
    ndcg_5_std: float
    precision_3_mean: float
    precision_3_std: float
    coverage_mean: float
    coverage_std: float


def kfold_cross_validate(
    model: Recommender,
    teams_df: pd.DataFrame,
    counter_matrix: pd.DataFrame,
    model_name: str,
    k: int = 5,
) -> CVResult:
    """K-fold cross-validation over the reconstruction team dataset.

    Splits the 250 teams into k folds. Each fold is held out as the test set
    while the remaining folds are used to compute per-fold metrics. Because the
    models use precomputed Smogon matrices (not the evaluation teams themselves),
    this does not retrain the model -- it gives stable metric estimates with
    variance across different subsets of teams.
    """
    team_ids = teams_df["team_id"].unique()
    np.random.seed(42)
    shuffled = np.random.permutation(team_ids)
    folds = np.array_split(shuffled, k)

    fold_metrics: dict[str, list[float]] = {
        "hit_rate_5": [], "ndcg_5": [], "precision_3": [], "coverage": []
    }

    for fold_ids in folds:
        fold_df = teams_df[teams_df["team_id"].isin(fold_ids)]
        result = evaluate_reconstruction(model, fold_df, counter_matrix, model_name=model_name)
        fold_metrics["hit_rate_5"].append(result.hit_rate_5)
        fold_metrics["ndcg_5"].append(result.ndcg_5)
        fold_metrics["precision_3"].append(result.precision_3)
        fold_metrics["coverage"].append(result.coverage)

    return CVResult(
        model_name=model_name,
        n_folds=k,
        hit_rate_5_mean=float(np.mean(fold_metrics["hit_rate_5"])),
        hit_rate_5_std=float(np.std(fold_metrics["hit_rate_5"])),
        ndcg_5_mean=float(np.mean(fold_metrics["ndcg_5"])),
        ndcg_5_std=float(np.std(fold_metrics["ndcg_5"])),
        precision_3_mean=float(np.mean(fold_metrics["precision_3"])),
        precision_3_std=float(np.std(fold_metrics["precision_3"])),
        coverage_mean=float(np.mean(fold_metrics["coverage"])),
        coverage_std=float(np.std(fold_metrics["coverage"])),
    )


def build_temporal_teams(artifacts_root: Path, regulation: str = "regh", top_n: int = 36) -> pd.DataFrame:
    """Build a leave-one-out evaluation dataset from a given regulation's usage data.

    Mirrors how reconstruction_teams.csv was built for RegG, but uses the
    specified regulation so we can evaluate temporal generalization.
    """
    usage_path = artifacts_root / "smogon" / f"{regulation}_usage.csv"
    teammates_path = artifacts_root / "smogon" / f"{regulation}_teammates.csv"

    if not usage_path.exists() or not teammates_path.exists():
        raise FileNotFoundError(
            f"Missing smogon data for regulation '{regulation}'. "
            f"Expected {usage_path} and {teammates_path}."
        )

    usage = pd.read_csv(usage_path)
    teammates = pd.read_csv(teammates_path)

    top_pokemon = (
        usage.sort_values("usage_pct", ascending=False)
        .head(top_n)["pokemon"]
        .tolist()
    )

    # Build teams: for each top Pokemon, find its top 5 most common teammates
    # that are also in the top pool to form realistic 6-mon teams
    rows = []
    team_id = 1
    for anchor in top_pokemon:
        anchor_mates = (
            teammates[teammates["pokemon"] == anchor]
            .sort_values("cooccur_pct", ascending=False)
        )
        partners = [
            t for t in anchor_mates["teammate"].tolist()
            if t in top_pokemon and t != anchor
        ][:5]
        if len(partners) < 5:
            extras = [p for p in top_pokemon if p != anchor and p not in partners]
            partners += extras[: 5 - len(partners)]
        team = [anchor] + partners
        for mon in team:
            rows.append({"team_id": team_id, "pokemon": mon, "regulation": regulation})
        team_id += 1

    return pd.DataFrame(rows)


def temporal_evaluate(
    model: Recommender,
    artifacts_root: Path,
    counter_matrix: pd.DataFrame,
    model_name: str,
    train_regulation: str = "regg",
    test_regulation: str = "regh",
) -> dict[str, object]:
    """Temporal split evaluation (CVTT).

    Evaluates a model trained on train_regulation data against teams built from
    test_regulation data. This measures whether the recommender generalises
    across competitive seasons rather than just fitting to one meta snapshot.
    """
    train_teams = pd.read_csv(
        artifacts_root / "eval" / "reconstruction_teams.csv"
    )
    train_result = evaluate_reconstruction(
        model, train_teams, counter_matrix, model_name=f"{model_name}_train"
    )

    test_teams = build_temporal_teams(artifacts_root, regulation=test_regulation)
    test_result = evaluate_reconstruction(
        model, test_teams, counter_matrix, model_name=f"{model_name}_test"
    )

    return {
        "model_name": model_name,
        "train_regulation": train_regulation,
        "test_regulation": test_regulation,
        "train": train_result.__dict__,
        "test": test_result.__dict__,
        "hit_rate_5_delta": test_result.hit_rate_5 - train_result.hit_rate_5,
        "ndcg_5_delta": test_result.ndcg_5 - train_result.ndcg_5,
    }


def write_eval_results(results: list[EvalResult], out_csv: Path, out_json: Path) -> None:
    rows = [r.__dict__ for r in results]
    write_csv(pd.DataFrame(rows), out_csv)
    write_json(rows, out_json)


def run_eval(cfg: ProjectConfig, models: dict[str, Recommender]) -> dict[str, Path]:
    teams = pd.read_csv(cfg.paths["artifacts_root"] / "eval" / "reconstruction_teams.csv")
    counter = pd.read_csv(cfg.paths["artifacts_root"] / "features" / "counter_matrix.csv").set_index("pokemon")
    eval_results: list[EvalResult] = []
    for name, model in models.items():
        eval_results.append(evaluate_reconstruction(model, teams, counter, model_name=name))
    out_csv = cfg.paths["reports_root"] / "offline_metrics.csv"
    out_json = cfg.paths["reports_root"] / "offline_metrics.json"
    write_eval_results(eval_results, out_csv, out_json)
    return {"csv": out_csv, "json": out_json}
