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
