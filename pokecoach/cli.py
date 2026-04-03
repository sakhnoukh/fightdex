from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pokecoach.config import ensure_dirs, load_config
from pokecoach.data_ingestion import (
    download_pokeapi_csv,
    download_smogon,
    parse_all_smogon,
    write_name_map,
    write_pokeapi_manifest,
    write_schema_manifest,
)
from pokecoach.evaluation import run_eval
from pokecoach.models import build_model_suite, load_model_data
from pokecoach.notebook_builder import build_notebook
from pokecoach.preprocess import run_preprocess
from pokecoach.simulation import run_simulation_sync, simulation_smoke_test_sync
from pokecoach.tuning import run_tuning
from pokecoach.utils import write_json


def cmd_download(config_path: str) -> None:
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    smogon_paths = download_smogon(cfg)
    pokeapi_paths = download_pokeapi_csv(cfg)
    parsed = parse_all_smogon(cfg)
    name_map = write_name_map(cfg)
    smogon_schema = write_schema_manifest(cfg)
    pokeapi_schema = write_pokeapi_manifest(cfg)
    print(f"Downloaded Smogon files: {len(smogon_paths)}")
    print(f"Downloaded PokeAPI files: {len(pokeapi_paths)}")
    print(f"Parsed artifacts: {len(parsed)}")
    print(f"Name map: {name_map}")
    print(f"Smogon schema manifest: {smogon_schema}")
    print(f"PokeAPI schema manifest: {pokeapi_schema}")


def cmd_preprocess(config_path: str) -> None:
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    outputs = run_preprocess(cfg)
    print("Preprocessing outputs:")
    for key, path in outputs.items():
        print(f"- {key}: {path}")


def _load_models(cfg_path: str):
    cfg = load_config(cfg_path)
    data = load_model_data(cfg.paths["artifacts_root"])
    models = build_model_suite(data)
    return cfg, models


def cmd_evaluate(config_path: str) -> None:
    cfg, models = _load_models(config_path)
    outputs = run_eval(cfg, models)
    print(f"Offline metrics CSV: {outputs['csv']}")
    print(f"Offline metrics JSON: {outputs['json']}")


def cmd_simulate(config_path: str, mode: str, tier: str) -> None:
    cfg, models = _load_models(config_path)
    smoke_ok = simulation_smoke_test_sync(tier=tier)
    print(f"Simulation smoke test passed: {smoke_ok}")
    outputs = run_simulation_sync(cfg, models, mode=mode, tier=tier)
    print(f"Simulation CSV: {outputs['csv']}")
    print(f"Simulation JSON: {outputs['json']}")


def cmd_demo_data(config_path: str) -> None:
    cfg = load_config(config_path)
    reports_root = cfg.paths["reports_root"]
    offline = reports_root / "offline_metrics.csv"
    sim = reports_root / "simulation_smoke.csv"
    payload = {"offline_metrics": [], "simulation_metrics": []}
    if offline.exists():
        payload["offline_metrics"] = pd.read_csv(offline).to_dict(orient="records")
    if sim.exists():
        payload["simulation_metrics"] = pd.read_csv(sim).to_dict(orient="records")
    out = reports_root / "demo_payload.json"
    write_json(payload, out)
    print(f"Demo payload: {out}")


def cmd_tune(config_path: str) -> None:
    cfg = load_config(config_path)
    ensure_dirs(cfg)
    outputs = run_tuning(cfg, verbose=True)
    print("Tuning outputs:")
    for key, path in outputs.items():
        print(f"- {key}: {path}")


def cmd_make_notebook(config_path: str) -> None:
    cfg = load_config(config_path)
    path = build_notebook(cfg)
    print(f"Notebook scaffold: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PokéCoach pipeline CLI")
    parser.add_argument("--config", default="configs/project.yaml", help="Path to config YAML.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("download", help="Download and parse raw data.")
    subparsers.add_parser("preprocess", help="Create feature artifacts.")
    subparsers.add_parser("evaluate", help="Run offline evaluation.")
    sim = subparsers.add_parser("simulate", help="Run poke-env simulations.")
    sim.add_argument("--mode", default="smoke", choices=["smoke", "dev", "final"])
    sim.add_argument("--tier", default="gen9vgc2024regg")
    subparsers.add_parser("tune", help="Run hyperparameter grid search.")
    subparsers.add_parser("demo-data", help="Build demo payload from metrics outputs.")
    subparsers.add_parser("make-notebook", help="Generate notebook scaffold.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "download":
        cmd_download(args.config)
    elif args.command == "preprocess":
        cmd_preprocess(args.config)
    elif args.command == "evaluate":
        cmd_evaluate(args.config)
    elif args.command == "simulate":
        cmd_simulate(args.config, args.mode, args.tier)
    elif args.command == "tune":
        cmd_tune(args.config)
    elif args.command == "demo-data":
        cmd_demo_data(args.config)
    elif args.command == "make-notebook":
        cmd_make_notebook(args.config)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
