from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProjectConfig:
    raw: dict[str, Any]
    root: Path

    @property
    def paths(self) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for key, rel in self.raw["paths"].items():
            paths[key] = self.root / rel
        return paths


def load_config(config_path: str | Path = "configs/project.yaml") -> ProjectConfig:
    path = Path(config_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    data = yaml.safe_load(path.read_text())
    return ProjectConfig(raw=data, root=Path.cwd())


def ensure_dirs(cfg: ProjectConfig) -> None:
    for directory in cfg.paths.values():
        directory.mkdir(parents=True, exist_ok=True)
