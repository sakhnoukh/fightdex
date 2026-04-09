from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from pokecoach.config import ProjectConfig
from pokecoach.utils import write_csv, write_json


def build_legal_pool(cfg: ProjectConfig, regulation: str = "regg") -> pd.DataFrame:
    usage = pd.read_csv(cfg.paths["artifacts_root"] / "smogon" / f"{regulation}_usage.csv")
    min_usage = float(cfg.raw["thresholds"]["min_usage_pct"])
    legal = usage[usage["usage_pct"] >= min_usage].copy()
    legal = legal.sort_values("usage_pct", ascending=False).drop_duplicates(subset=["pokemon"])
    return legal.reset_index(drop=True)


def build_cooccurrence_matrix(legal: pd.DataFrame, teammates: pd.DataFrame) -> pd.DataFrame:
    names = legal["pokemon"].tolist()
    matrix = pd.DataFrame(0.0, index=names, columns=names)
    filtered = teammates[teammates["pokemon"].isin(names) & teammates["teammate"].isin(names)]
    for _, row in filtered.iterrows():
        p1, p2 = row["pokemon"], row["teammate"]
        score = float(row["cooccur_pct"]) / 100.0
        matrix.loc[p1, p2] = max(matrix.loc[p1, p2], score)
        matrix.loc[p2, p1] = max(matrix.loc[p2, p1], score)
    diag = matrix.to_numpy(copy=True)
    np.fill_diagonal(diag, 1.0)
    matrix[:] = diag
    return matrix


def _build_types_lookup(cfg: ProjectConfig) -> dict[str, list[str]]:
    pokemon = pd.read_csv(cfg.paths["pokeapi_root"] / "pokemon.csv")
    ptypes = pd.read_csv(cfg.paths["pokeapi_root"] / "pokemon_types.csv")
    types = pd.read_csv(cfg.paths["pokeapi_root"] / "types.csv")
    merged = ptypes.merge(types, left_on="type_id", right_on="id", suffixes=("", "_type"))
    merged = merged.merge(pokemon[["id", "identifier"]], left_on="pokemon_id", right_on="id", suffixes=("", "_poke"))
    out: dict[str, list[str]] = {}
    for name, group in merged.groupby("identifier_poke"):
        out[name] = group.sort_values("slot")["identifier"].tolist()
    return out


def build_pokemon_types(cfg: ProjectConfig, legal: pd.DataFrame) -> dict[str, list[str]]:
    types_lookup = _build_types_lookup(cfg)
    result: dict[str, list[str]] = {}
    for name in legal["pokemon"]:
        normalized = name.lower().replace(" ", "-").replace(".", "").replace("'", "")
        result[name] = types_lookup.get(normalized, [])
    return result


def build_content_features(cfg: ProjectConfig, legal: pd.DataFrame) -> pd.DataFrame:
    pokemon = pd.read_csv(cfg.paths["pokeapi_root"] / "pokemon.csv")
    pstats = pd.read_csv(cfg.paths["pokeapi_root"] / "pokemon_stats.csv")
    stats = pd.read_csv(cfg.paths["pokeapi_root"] / "stats.csv")
    merged = pstats.merge(stats, left_on="stat_id", right_on="id", suffixes=("", "_stat"))
    pivot = merged.pivot_table(index="pokemon_id", columns="identifier", values="base_stat", fill_value=0)
    stats_with_name = pivot.merge(
        pokemon[["id", "identifier"]],
        left_index=True,
        right_on="id",
        how="left",
    )
    stats_with_name = stats_with_name.rename(columns={"identifier": "pokemon"}).drop(columns=["id"])
    legal_set = set(legal["pokemon"].str.lower().str.replace(" ", "-", regex=False))
    stats_with_name = stats_with_name[stats_with_name["pokemon"].isin(legal_set)]
    return stats_with_name.reset_index(drop=True)


def build_counter_matrix(cfg: ProjectConfig, legal: pd.DataFrame) -> pd.DataFrame:
    types_lookup = _build_types_lookup(cfg)
    type_efficacy = pd.read_csv(cfg.paths["pokeapi_root"] / "type_efficacy.csv")
    types = pd.read_csv(cfg.paths["pokeapi_root"] / "types.csv")[["id", "identifier"]]
    type_id = {r["identifier"]: int(r["id"]) for _, r in types.iterrows()}

    legal_names = legal["pokemon"].tolist()
    normalized_names = {name: name.lower().replace(" ", "-").replace(".", "").replace("'", "") for name in legal_names}
    matrix = pd.DataFrame(0.0, index=legal_names, columns=legal_names)

    def matchup(attacker: str, defender: str) -> float:
        atk_types = types_lookup.get(attacker, [])
        def_types = types_lookup.get(defender, [])
        if not atk_types or not def_types:
            return 1.0
        best = 1.0
        for atk in atk_types:
            mult = 1.0
            for d in def_types:
                row = type_efficacy[
                    (type_efficacy["damage_type_id"] == type_id.get(atk))
                    & (type_efficacy["target_type_id"] == type_id.get(d))
                ]
                if row.empty:
                    continue
                mult *= float(row.iloc[0]["damage_factor"]) / 100.0
            best = max(best, mult)
        return best

    for p1 in legal_names:
        for p2 in legal_names:
            matrix.loc[p1, p2] = matchup(normalized_names[p1], normalized_names[p2])
    return matrix


def _build_ability_lookup(data_root: Path) -> dict[str, str]:
    """Returns {pokeapi_identifier: showdown_ability_name} for slot-1 abilities."""
    pokemon_csv = pd.read_csv(data_root / "pokeapi" / "pokemon.csv")
    abilities_csv = pd.read_csv(data_root / "pokeapi" / "abilities.csv")
    pokemon_abilities_csv = pd.read_csv(data_root / "pokeapi" / "pokemon_abilities.csv")

    poke_id_to_name = dict(zip(pokemon_csv["id"], pokemon_csv["identifier"]))
    ability_id_to_name = dict(zip(abilities_csv["id"], abilities_csv["identifier"]))

    lookup: dict[str, str] = {}
    slot1 = pokemon_abilities_csv[
        (pokemon_abilities_csv["slot"] == 1) &
        (pokemon_abilities_csv["is_hidden"] == 0)
    ]
    for _, row in slot1.iterrows():
        pname = poke_id_to_name.get(row["pokemon_id"], "")
        aname = ability_id_to_name.get(row["ability_id"], "")
        if pname and aname:
            # Convert pokeapi format (inner-focus) → Showdown format (Inner Focus)
            lookup[pname] = aname.replace("-", " ").title()
    return lookup


# Competitive ability overrides for VGC — covers Pokémon missing from the PokeAPI
# slot-1 lookup (which would incorrectly fall back to "Pressure") and Pokémon
# whose slot-1 ability is legal but non-competitive.
_COMPETITIVE_ABILITIES: dict[str, str] = {
    # ── Missing from PokeAPI lookup (would get fallback "Pressure") ──────────
    "tornadus": "Prankster",
    "thundurus": "Prankster",
    "landorus": "Intimidate",
    "indeedee-f": "Psychic Surge",
    "tatsugiri": "Commander",
    "maushold": "Friend Guard",
    "mimikyu": "Disguise",
    "urshifu": "Unseen Fist",
    "ogerpon-wellspring": "Water Absorb",
    "ogerpon-cornerstone": "Sturdy",
    "ogerpon-hearthflame": "Mold Breaker",
    "necrozma-dawn-wings": "Prism Armor",
    "necrozma-dusk-mane": "Prism Armor",
    "basculegion": "Swift Swim",
    # ── Slot-1 is non-competitive — override to the used competitive ability ──
    "incineroar": "Intimidate",
    "rillaboom": "Grassy Surge",
    "pelipper": "Drizzle",
    "politoed": "Drizzle",
    "amoonguss": "Regenerator",
    "torkoal": "Drought",
    "ninetales-alola": "Snow Warning",
    "hippowdon": "Sand Stream",
    "tyranitar": "Sand Stream",
    "excadrill": "Sand Rush",
    "whimsicott": "Prankster",
    "grimmsnarl": "Prankster",
    "porygon2": "Download",
    "dusclops": "Frisk",
    "gothitelle": "Shadow Tag",
    "amoonguss": "Regenerator",
    "hatterene": "Magic Bounce",
    "aromatisse": "Aroma Veil",
    "sylveon": "Pixilate",
    "togekiss": "Serene Grace",
}


def build_canonical_pastes(
    moveset_df: pd.DataFrame,
    ability_lookup: dict[str, str] | None = None,
    top_moves: int = 4,
) -> dict[str, str]:
    pastes: dict[str, str] = {}
    for name, group in moveset_df.groupby("pokemon"):
        moves = (
            group.sort_values("usage_pct", ascending=False)["move"]
            .head(top_moves)
            .astype(str)
            .tolist()
        )
        if not moves:
            continue
        norm = name.lower().replace(" ", "-").replace(".", "").replace("'", "")
        ability = (
            _COMPETITIVE_ABILITIES.get(norm)
            or (ability_lookup or {}).get(norm)
            or "Pressure"
        )
        lines = [f"{name} @ Leftovers", f"Ability: {ability}", "Tera Type: Normal", "EVs: 252 HP / 252 Atk / 4 Spe", "Adamant Nature"]
        lines.extend([f"- {m}" for m in moves])
        pastes[name] = "\n".join(lines)
    return pastes


def build_reconstruction_dataset(cfg: ProjectConfig) -> pd.DataFrame:
    usage = pd.read_csv(cfg.paths["artifacts_root"] / "smogon" / "regh_usage.csv")
    top = usage.sort_values("usage_pct", ascending=False).drop_duplicates(subset=["pokemon"]).head(120)
    mons = top["pokemon"].tolist()
    teams: list[dict[str, object]] = []
    team_id = 0
    for combo in combinations(mons[:36], 6):
        team_id += 1
        for mon in combo:
            teams.append({"team_id": team_id, "pokemon": mon, "regulation": "regh"})
        if team_id >= 250:
            break
    return pd.DataFrame(teams)


def run_preprocess(cfg: ProjectConfig) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    legal = build_legal_pool(cfg, regulation="regg")
    teammates = pd.read_csv(cfg.paths["artifacts_root"] / "smogon" / "regg_teammates.csv")
    moveset = pd.read_csv(cfg.paths["artifacts_root"] / "smogon" / "regg_moveset.csv")

    cooc = build_cooccurrence_matrix(legal, teammates)
    content = build_content_features(cfg, legal)
    counter = build_counter_matrix(cfg, legal)
    pokemon_types = build_pokemon_types(cfg, legal)
    ability_lookup = _build_ability_lookup(cfg.paths["data_root"])
    pastes = build_canonical_pastes(moveset, ability_lookup=ability_lookup)
    reconstruction = build_reconstruction_dataset(cfg)

    legal_out = cfg.paths["artifacts_root"] / "features" / "legal_pool.csv"
    cooc_out = cfg.paths["artifacts_root"] / "features" / "cooccurrence.csv"
    content_out = cfg.paths["artifacts_root"] / "features" / "content_features.csv"
    counter_out = cfg.paths["artifacts_root"] / "features" / "counter_matrix.csv"
    pastes_out = cfg.paths["artifacts_root"] / "features" / "canonical_pastes.json"
    reconstruction_out = cfg.paths["artifacts_root"] / "eval" / "reconstruction_teams.csv"
    types_out = cfg.paths["artifacts_root"] / "features" / "pokemon_types.json"

    write_csv(legal, legal_out)
    write_csv(cooc.reset_index().rename(columns={"index": "pokemon"}), cooc_out)
    write_csv(content, content_out)
    write_csv(counter.reset_index().rename(columns={"index": "pokemon"}), counter_out)
    write_json(pokemon_types, types_out)
    write_json(pastes, pastes_out)
    write_csv(reconstruction, reconstruction_out)

    outputs["legal_pool"] = legal_out
    outputs["cooccurrence"] = cooc_out
    outputs["content"] = content_out
    outputs["counter"] = counter_out
    outputs["pokemon_types"] = types_out
    outputs["pastes"] = pastes_out
    outputs["reconstruction"] = reconstruction_out
    return outputs
