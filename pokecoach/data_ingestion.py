from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from pokecoach.config import ProjectConfig
from pokecoach.utils import write_csv, write_json


def _download_text(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.write_text(response.text)


def _download_with_fallback(urls: list[str], destination: Path) -> str:
    errors: list[str] = []
    for url in urls:
        try:
            _download_text(url, destination)
            return url
        except Exception as exc:
            errors.append(f"{url} => {exc}")
    raise RuntimeError("All download URLs failed:\n" + "\n".join(errors))


def download_smogon(cfg: ProjectConfig) -> list[Path]:
    smogon_cfg = cfg.raw["smogon"]
    out_paths: list[Path] = []
    smogon_root = cfg.paths["smogon_root"]
    for reg, months in smogon_cfg["months"].items():
        for month in months:
            reg_letter = reg[-1]
            usage_url = smogon_cfg["usage_base_url"].format(month=month, reg=reg_letter)
            moveset_url = smogon_cfg["moveset_base_url"].format(month=month, reg=reg_letter)
            usage_out = smogon_root / reg / f"{month}_usage.txt"
            moveset_out = smogon_root / reg / f"{month}_moveset.txt"
            usage_alts = [
                usage_url,
                usage_url.replace("-0.txt", "-1760.txt"),
                usage_url.replace("-0.txt", "-1630.txt"),
            ]
            moveset_alts = [
                moveset_url,
                moveset_url.replace("-0.txt", "-1760.txt"),
                moveset_url.replace("-0.txt", "-1630.txt"),
            ]
            _download_with_fallback(usage_alts, usage_out)
            _download_with_fallback(moveset_alts, moveset_out)
            out_paths.extend([usage_out, moveset_out])
    return out_paths


def download_pokeapi_csv(cfg: ProjectConfig) -> list[Path]:
    pokeapi_cfg = cfg.raw["pokeapi"]
    out_paths: list[Path] = []
    root = cfg.paths["pokeapi_root"]
    for file_name in pokeapi_cfg["files"]:
        url = f"{pokeapi_cfg['csv_base_url']}/{file_name}"
        out_path = root / file_name
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(response.content)
        out_paths.append(out_path)
    return out_paths


def write_pokeapi_manifest(cfg: ProjectConfig) -> Path:
    root = cfg.paths["pokeapi_root"]
    manifest: dict[str, dict[str, object]] = {}
    for path in sorted(root.glob("*.csv")):
        frame = pd.read_csv(path)
        manifest[path.name] = {
            "rows": int(len(frame)),
            "columns": frame.columns.tolist(),
        }
    out = cfg.paths["artifacts_root"] / "schemas" / "pokeapi_schema_manifest.json"
    write_json(manifest, out)
    return out


_SECTION_SKIP: frozenset[str] = frozenset(
    {"Abilities", "Items", "Spreads", "Moves", "Teammates", "Checks and Counters"}
)


def _clean_name(raw: str) -> str:
    name = raw.strip()
    name = name.replace("%", "")
    return re.sub(r"\s+", " ", name)


def _is_pokemon_header(name: str) -> bool:
    """Return True only if name looks like a Pokemon name, not a section header or stat line."""
    if name in _SECTION_SKIP:
        return False
    if ":" in name:  # "Raw count: 883180", "Avg. weight: 1.0"
        return False
    if any(c.isdigit() for c in name):  # "Viability Ceiling: 83"
        return False
    return bool(name)


def parse_usage_file(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pattern = re.compile(r"^\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([\d.]+)%\s*\|")
    for line in path.read_text().splitlines():
        match = pattern.search(line)
        if not match:
            continue
        rank, name, usage = match.groups()
        rows.append(
            {
                "rank": int(rank),
                "pokemon": _clean_name(name),
                "usage_pct": float(usage),
            }
        )
    return pd.DataFrame(rows)


def parse_teammates_file(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    current: str | None = None
    header_pattern = re.compile(r"^\s*\|\s*([A-Za-z0-9\-\s\.'’:]+)\s*\|\s*$")
    teammate_pattern = re.compile(r"^\s*\|\s*([A-Za-z0-9\-\s\.'()/:]+?)\s+([\d.]+)%\s*\|?\s*$")
    in_teammates = False

    for line in path.read_text().splitlines():
        if "Teammates" in line:
            in_teammates = True
            continue
        if "Checks and Counters" in line or "Moves" in line:
            in_teammates = False
        if line.startswith("+-"):
            continue
        if line.strip().startswith("|") and "%" not in line and not in_teammates:
            header_match = header_pattern.search(line)
            if header_match:
                cleaned = _clean_name(header_match.group(1))
                if _is_pokemon_header(cleaned):
                    current = cleaned
            continue
        if in_teammates and current:
            teammate_match = teammate_pattern.search(line)
            if teammate_match:
                teammate, pct = teammate_match.groups()
                rows.append(
                    {
                        "pokemon": current,
                        "teammate": _clean_name(teammate),
                        "cooccur_pct": float(pct),
                    }
                )
    return pd.DataFrame(rows)


def parse_moveset_file(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    current: str | None = None
    in_moves = False
    header_pattern = re.compile(r"^\s*\|\s*([A-Za-z0-9\-\s\.'’:]+)\s*\|\s*$")
    move_pattern = re.compile(r"^\s*\|\s*([A-Za-z0-9\-\s\.'/()]+?)\s+([\d.]+)%\s*\|?\s*$")
    for line in path.read_text().splitlines():
        if "Moves" in line:
            in_moves = True
            continue
        if in_moves and ("Abilities" in line or "Items" in line or "Teammates" in line or "Checks and Counters" in line):
            in_moves = False
        if line.startswith("+-"):
            continue
        if line.strip().startswith("|") and "%" not in line and not in_moves:
            m = header_pattern.search(line)
            if m:
                cleaned = _clean_name(m.group(1))
                if _is_pokemon_header(cleaned):
                    current = cleaned
            continue
        if in_moves and current:
            mm = move_pattern.search(line)
            if mm:
                move, pct = mm.groups()
                rows.append(
                    {
                        "pokemon": current,
                        "move": _clean_name(move),
                        "usage_pct": float(pct),
                    }
                )
    return pd.DataFrame(rows)


def parse_all_smogon(cfg: ProjectConfig) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for reg in ("regg", "regh"):
        reg_dir = cfg.paths["smogon_root"] / reg
        usage_frames: list[pd.DataFrame] = []
        teammate_frames: list[pd.DataFrame] = []
        moveset_frames: list[pd.DataFrame] = []
        for usage_path in sorted(reg_dir.glob("*_usage.txt")):
            usage_frames.append(parse_usage_file(usage_path).assign(regulation=reg, source=usage_path.name))
        for moveset_path in sorted(reg_dir.glob("*_moveset.txt")):
            teammate_frames.append(parse_teammates_file(moveset_path).assign(regulation=reg, source=moveset_path.name))
            moveset_frames.append(parse_moveset_file(moveset_path).assign(regulation=reg, source=moveset_path.name))

        usage_df = pd.concat(usage_frames, ignore_index=True) if usage_frames else pd.DataFrame()
        teammates_df = pd.concat(teammate_frames, ignore_index=True) if teammate_frames else pd.DataFrame()
        moveset_df = pd.concat(moveset_frames, ignore_index=True) if moveset_frames else pd.DataFrame()

        usage_out = cfg.paths["artifacts_root"] / "smogon" / f"{reg}_usage.csv"
        teammates_out = cfg.paths["artifacts_root"] / "smogon" / f"{reg}_teammates.csv"
        moveset_out = cfg.paths["artifacts_root"] / "smogon" / f"{reg}_moveset.csv"
        _validate_columns(usage_df, ["rank", "pokemon", "usage_pct", "regulation", "source"], f"{reg}_usage")
        _validate_columns(teammates_df, ["pokemon", "teammate", "cooccur_pct", "regulation", "source"], f"{reg}_teammates")
        _validate_columns(moveset_df, ["pokemon", "move", "usage_pct", "regulation", "source"], f"{reg}_moveset")
        write_csv(usage_df, usage_out)
        write_csv(teammates_df, teammates_out)
        write_csv(moveset_df, moveset_out)
        outputs[f"{reg}_usage"] = usage_out
        outputs[f"{reg}_teammates"] = teammates_out
        outputs[f"{reg}_moveset"] = moveset_out
    return outputs


def create_name_map(names: Iterable[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in names:
        canonical = name.lower().replace(".", "").replace("'", "")
        canonical = canonical.replace(" ", "-")
        mapping[name] = canonical
    return mapping


def write_name_map(cfg: ProjectConfig) -> Path:
    usage = pd.read_csv(cfg.paths["artifacts_root"] / "smogon" / "regg_usage.csv")
    names = sorted(set(usage["pokemon"].dropna().tolist()))
    mapping = create_name_map(names)
    out = cfg.paths["artifacts_root"] / "mappings" / "pokemon_name_map.json"
    write_json(mapping, out)
    return out


def _validate_columns(df: pd.DataFrame, required: list[str], artifact_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{artifact_name} missing columns: {missing}")
    if df.empty:
        raise ValueError(f"{artifact_name} is empty after parsing")


def write_schema_manifest(cfg: ProjectConfig) -> Path:
    root = cfg.paths["artifacts_root"] / "smogon"
    manifest: dict[str, dict[str, object]] = {}
    for path in sorted(root.glob("*.csv")):
        frame = pd.read_csv(path)
        manifest[path.name] = {
            "rows": int(len(frame)),
            "columns": frame.columns.tolist(),
        }
    out = cfg.paths["artifacts_root"] / "schemas" / "smogon_schema_manifest.json"
    write_json(manifest, out)
    return out
