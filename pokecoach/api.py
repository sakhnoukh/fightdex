"""Clean public interface for team recommendations."""
from __future__ import annotations

from pokecoach.config import load_config
from pokecoach.models import HybridRecommender, ModelData, load_model_data, recommend_moveset

_model_data: ModelData | None = None
_hybrid: HybridRecommender | None = None


def _get_model() -> tuple[HybridRecommender, ModelData]:
    global _model_data, _hybrid
    if _model_data is None:
        cfg = load_config()
        _model_data = load_model_data(cfg.paths["artifacts_root"])
        _hybrid = HybridRecommender(_model_data)
    return _hybrid, _model_data  # type: ignore[return-value]


def get_legal_pool() -> list[str]:
    """Return all pokemon names in the legal pool, sorted by usage."""
    _, data = _get_model()
    return data.legal_pool.sort_values("usage_pct", ascending=False)["pokemon"].tolist()


def recommend_team(
    partial_team: list[str],
    opponent_context: list[str] | None = None,
    type_preferences: list[str] | None = None,
    weight_overrides: dict | None = None,
    n_recommendations: int | None = None,
) -> dict:
    """
    Recommend Pokemon to complete a team.

    Args:
        partial_team: Pokemon already chosen for your team (0–6).
        opponent_context: Known opponent Pokemon (optional).
        type_preferences: Restrict recommendations to these types, e.g. ["water", "grass"].
        weight_overrides: Override scoring weights, e.g. {"synergy": 0.6, "counter": 0.1,
                          "viability": 0.2, "content": 0.1}.
        n_recommendations: How many to return. Defaults to filling a 6-mon team.

    Returns:
        {
          "recommendations": [{
            "pokemon": str, "moves": list[str], "types": list[str],
            "scores": {"synergy": float, "viability": float, "counter": float},
            "synergy_partners": [(name, value), ...],
            "beats": [(opp_name, multiplier), ...],
          }, ...],
          "partial_team": list[str],
          "opponent_context": list[str],
        }
    """
    hybrid, data = _get_model()

    if n_recommendations is None:
        n_recommendations = max(0, 6 - len(partial_team))

    if n_recommendations == 0:
        return {
            "recommendations": [],
            "partial_team": partial_team,
            "opponent_context": opponent_context or [],
        }

    recs = hybrid.recommend(
        partial_team,
        opponent_context=opponent_context,
        k=n_recommendations,
        type_filter=type_preferences,
        weight_overrides=weight_overrides,
    )

    max_usage = float(data.legal_pool["usage_pct"].max()) or 1.0
    # Cooccurrence is already 0-1 normalised (max=1.0), but clamp defensively.
    max_cooc = float(data.cooccurrence.values.max()) or 1.0

    results = []
    for mon in recs:
        # --- Synergy: max cooccurrence with any existing team member ---
        cooc_vals: list[tuple[str, float]] = []
        if partial_team:
            for t in partial_team:
                if t in data.cooccurrence.index and mon in data.cooccurrence.columns:
                    cooc_vals.append((t, float(data.cooccurrence.loc[t, mon])))
            cooc_vals.sort(key=lambda x: x[1], reverse=True)
        synergy_norm = min(1.0, cooc_vals[0][1] / max_cooc) if cooc_vals else 0.0
        synergy_partners = cooc_vals[:2]

        # --- Viability: usage_pct / max_usage_pct ---
        mon_row = data.legal_pool[data.legal_pool["pokemon"] == mon]
        viability = float(mon_row["usage_pct"].iloc[0]) / max_usage if not mon_row.empty else 0.0

        # --- Counter: mean effectiveness vs opponent / 4.0 ---
        beats: list[tuple[str, float]] = []
        if opponent_context and mon in data.counter.index:
            counter_vals = []
            for opp in opponent_context:
                if opp in data.counter.columns:
                    val = float(data.counter.loc[mon, opp])
                    counter_vals.append(val)
                    if val > 1.0:
                        beats.append((opp, val))
            counter_score = min(1.0, (sum(counter_vals) / max(1, len(counter_vals))) / 4.0)
        else:
            counter_score = 0.0

        results.append(
            {
                "pokemon": mon,
                "moves": recommend_moveset(mon, data.moveset, k=4),
                "types": data.pokemon_types.get(mon, []),
                "scores": {
                    "synergy": round(synergy_norm, 3),
                    "viability": round(viability, 3),
                    "counter": round(counter_score, 3),
                },
                "synergy_partners": synergy_partners,
                "beats": beats,
            }
        )

    return {
        "recommendations": results,
        "partial_team": partial_team,
        "opponent_context": opponent_context or [],
    }


def get_matchup_matrix(your_team: list[str], opp_team: list[str]) -> dict:
    """
    Return a type-effectiveness matrix of your_team vs opp_team.

    Returns:
        {
          "your_team": [...], "opp_team": [...],
          "matrix": {"Incineroar": {"Rillaboom": 2.0, ...}, ...}
        }
    """
    _, data = _get_model()
    matrix: dict[str, dict[str, float]] = {}
    for ym in your_team:
        matrix[ym] = {}
        for om in opp_team:
            if ym in data.counter.index and om in data.counter.columns:
                matrix[ym][om] = float(data.counter.loc[ym, om])
            else:
                matrix[ym][om] = 1.0
    return {"your_team": your_team, "opp_team": opp_team, "matrix": matrix}


def get_win_probability(your_team: list[str], opp_team: list[str]) -> dict:
    """
    Analytical win probability estimate based on counter matrix.

    Formula:
        your_score = mean(counter[ym, om] for all (ym, om) pairs)
        opp_score  = mean(counter[om, ym] for all (om, ym) pairs)
        win_prob   = your_score / (your_score + opp_score)

    Returns:
        {"win_probability": float, "your_coverage": float, "opp_coverage": float}
    """
    _, data = _get_model()
    your_scores: list[float] = []
    opp_scores: list[float] = []
    for ym in your_team:
        for om in opp_team:
            if ym in data.counter.index and om in data.counter.columns:
                your_scores.append(float(data.counter.loc[ym, om]))
            if om in data.counter.index and ym in data.counter.columns:
                opp_scores.append(float(data.counter.loc[om, ym]))
    ys = sum(your_scores) / max(1, len(your_scores)) if your_scores else 1.0
    os_ = sum(opp_scores) / max(1, len(opp_scores)) if opp_scores else 1.0
    win_prob = ys / (ys + os_) if (ys + os_) > 0 else 0.5
    return {
        "win_probability": round(win_prob, 4),
        "your_coverage": round(ys, 4),
        "opp_coverage": round(os_, 4),
    }


def run_battle_sim(your_team: list[str], opp_team: list[str], n: int = 400) -> dict:
    """Run simulated battles; falls back to analytical if Showdown is offline."""
    from pokecoach.simulation import run_team_sim

    return run_team_sim(your_team, opp_team, n=n)
