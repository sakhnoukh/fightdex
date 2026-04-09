from __future__ import annotations

import asyncio
import json
import math
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from pokecoach.config import ProjectConfig
from pokecoach.models import BaseModel
from pokecoach.utils import write_csv, write_json

try:
    from poke_env import AccountConfiguration
    from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
except Exception:  # pragma: no cover
    AccountConfiguration = None
    RandomPlayer = None
    SimpleHeuristicsPlayer = None


@dataclass
class SimulationResult:
    model_name: str
    wins: int
    total: int

    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.total)

    @property
    def ci95(self) -> float:
        p = self.win_rate
        n = max(1, self.total)
        return 1.96 * math.sqrt((p * (1 - p)) / n)


def check_showdown_running(host: str = "localhost", port: int = 8000) -> bool:
    """Return True if a Showdown server is reachable at host:port."""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


_ITEM_POOL = [
    "Leftovers", "Life Orb", "Focus Sash", "Assault Vest", "Choice Scarf",
    "Choice Specs", "Choice Band", "Lum Berry", "Sitrus Berry", "Rocky Helmet",
    "Booster Energy", "Clear Amulet", "Safety Goggles", "Wide Lens", "Power Herb",
]


def _build_team_text(team: list[str], pastes: dict[str, str]) -> str:
    chunks = []
    used_items: set[str] = set()
    for mon in team:
        paste = pastes.get(mon, f"{mon}\n- Protect\n- Taunt\n- U-turn\n- Helping Hand")
        # Assign a unique item to satisfy VGC Item Clause (no duplicate held items)
        for item in _ITEM_POOL:
            if item not in used_items:
                lines = paste.split("\n")
                if " @ " in lines[0]:
                    lines[0] = lines[0].split(" @ ")[0] + f" @ {item}"
                paste = "\n".join(lines)
                used_items.add(item)
                break
        chunks.append(paste)
    return "\n\n".join(chunks)



async def _run_battles_persistent(
    your_text: str, opp_text: str, tier: str, n: int, batch_size: int = 50
) -> int:
    if SimpleHeuristicsPlayer is None:
        raise RuntimeError("poke_env not installed")
    import uuid
    from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
    run_id = uuid.uuid4().hex[:8]
    player = SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration(f"Coach_{run_id}_p1", None),
        team=your_text,
        battle_format=tier,
        server_configuration=LocalhostServerConfiguration,
    )
    opponent = SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration(f"Coach_{run_id}_p2", None),
        team=opp_text,
        battle_format=tier,
        server_configuration=LocalhostServerConfiguration,
    )
    try:
        for batch_start in range(0, n, batch_size):
            batch = min(batch_size, n - batch_start)
            await player.battle_against(opponent, n_battles=batch)
            done = batch_start + batch
            wins = player.n_won_battles
            print(f"  Battles: {done}/{n} — wins: {wins} ({wins/done*100:.1f}%)", flush=True)
    finally:
        await player.ps_client.stop_listening()
        await opponent.ps_client.stop_listening()
    return player.n_won_battles


def _analytical_win_prob(your_team: list[str], opp_team: list[str], counter: pd.DataFrame) -> float:
    """Estimate win probability from type-effectiveness counter matrix."""
    your_scores: list[float] = []
    opp_scores: list[float] = []
    for ym in your_team:
        for om in opp_team:
            if ym in counter.index and om in counter.columns:
                your_scores.append(float(counter.loc[ym, om]))
            if om in counter.index and ym in counter.columns:
                opp_scores.append(float(counter.loc[om, ym]))
    ys = sum(your_scores) / max(1, len(your_scores)) if your_scores else 1.0
    os_ = sum(opp_scores) / max(1, len(opp_scores)) if opp_scores else 1.0
    return ys / (ys + os_) if (ys + os_) > 0 else 0.5


def run_team_sim(your_team: list[str], opp_team: list[str], n: int = 400) -> dict:
    """
    Run N battles between your_team and opp_team.

    Uses real Showdown simulation if the server is running, otherwise falls
    back to an analytical estimate from the counter matrix.

    Returns:
        {"wins": int, "total": int, "win_rate": float, "used_simulation": bool}
    """
    from pokecoach.config import load_config

    cfg = load_config()
    pastes_path = cfg.paths["artifacts_root"] / "features" / "canonical_pastes.json"
    pastes: dict[str, str] = {}
    if pastes_path.exists():
        pastes = json.loads(pastes_path.read_text())

    if check_showdown_running():
        tier = cfg.raw["regulation"]["showdown_tier"]
        your_text = _build_team_text(your_team, pastes)
        opp_text = _build_team_text(opp_team, pastes)
        try:
            wins = asyncio.run(_run_battles_persistent(your_text, opp_text, tier, n))
            return {"wins": wins, "total": n, "win_rate": wins / max(1, n), "used_simulation": True}
        except Exception as e:
            print(f"Simulation error: {e}", flush=True)
            # fall through to analytical

    # Analytical fallback
    counter_path = cfg.paths["artifacts_root"] / "features" / "counter_matrix.csv"
    counter = pd.read_csv(counter_path).set_index("pokemon")
    win_prob = _analytical_win_prob(your_team, opp_team, counter)
    wins = round(win_prob * n)
    return {"wins": wins, "total": n, "win_rate": win_prob, "used_simulation": False}


def _sample_base_teams(teams_df: pd.DataFrame, max_teams: int, max_size: int = 5) -> list[list[str]]:
    base_teams: list[list[str]] = []
    grouped = teams_df.groupby("team_id")["pokemon"].apply(list).tolist()
    for team in grouped[:max_teams]:
        if len(team) >= 5:
            base_teams.append(team[:max_size])
    return base_teams


async def evaluate_model_simulation(
    model_name: str,
    model: BaseModel,
    cfg: ProjectConfig,
    mode: str = "smoke",
    tier: str | None = None,
) -> SimulationResult:
    sizes = cfg.raw["simulation"]["sizes"][mode]
    teams = pd.read_csv(cfg.paths["artifacts_root"] / "eval" / "reconstruction_teams.csv")
    pastes = {}
    pastes_path = cfg.paths["artifacts_root"] / "features" / "canonical_pastes.json"
    if pastes_path.exists():
        pastes = json.loads(pastes_path.read_text())

    # Treat each sampled tournament team as the opponent (full 6 mons)
    opp_teams = _sample_base_teams(teams, sizes["teams"], max_size=6)
    total = 0
    wins = 0
    format_tier = tier or cfg.raw["regulation"]["showdown_tier"]

    for opp_team in opp_teams:
        # Recommend a full counter-team against this specific opponent
        rec = model.recommend([], opponent_context=opp_team, k=6)
        if not rec:
            continue
        rec_text = _build_team_text(rec, pastes)
        opp_text = _build_team_text(opp_team, pastes)
        n_battles = sizes["battles_per_team"]
        result_wins = await _run_battles_persistent(rec_text, opp_text, format_tier, n_battles)
        wins += result_wins
        total += n_battles
        if hasattr(model, "update_reward"):
            for mon in rec:
                model.update_reward(mon, result_wins / max(1, n_battles))
    return SimulationResult(model_name=model_name, wins=wins, total=total)


async def run_simulation(
    cfg: ProjectConfig,
    models: dict[str, BaseModel],
    mode: str = "smoke",
    tier: str | None = None,
) -> dict[str, Path]:
    rows: list[dict[str, Any]] = []
    for name, model in models.items():
        result = await evaluate_model_simulation(name, model, cfg, mode=mode, tier=tier)
        rows.append(
            {
                "model": name,
                "wins": result.wins,
                "battles": result.total,
                "win_rate": result.win_rate,
                "ci95": result.ci95,
                "mode": mode,
                "tier": tier or cfg.raw["regulation"]["showdown_tier"],
            }
        )
    out_csv = cfg.paths["reports_root"] / f"simulation_{mode}.csv"
    out_json = cfg.paths["reports_root"] / f"simulation_{mode}.json"
    write_csv(pd.DataFrame(rows), out_csv)
    write_json(rows, out_json)
    return {"csv": out_csv, "json": out_json}


def run_simulation_sync(
    cfg: ProjectConfig,
    models: dict[str, BaseModel],
    mode: str = "smoke",
    tier: str | None = None,
) -> dict[str, Path]:
    return asyncio.run(run_simulation(cfg, models, mode=mode, tier=tier))


async def simulation_smoke_test(tier: str = "gen9vgc2024regg") -> bool:
    if RandomPlayer is None:
        return False
    p1 = RandomPlayer(battle_format=tier)
    p2 = RandomPlayer(battle_format=tier)
    await p1.battle_against(p2, n_battles=1)
    return True


def simulation_smoke_test_sync(tier: str = "gen9vgc2024regg") -> bool:
    try:
        return asyncio.run(simulation_smoke_test(tier))
    except Exception:
        return False
