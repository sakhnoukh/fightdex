from __future__ import annotations

import json
from pathlib import Path

from pokecoach.config import ProjectConfig


SECTION_TITLES = [
    "1. Data Loading and Parsing",
    "2. Exploratory Data Analysis",
    "3. Data Preparation and Feature Engineering",
    "4. Building the Simulation Ground Truth",
    "5. Evaluation Framework and Metrics",
    "6. Baseline 1 - Random Recommender",
    "7. Baseline 2 - Popularity and Collaborative Filtering",
    "8. Content-Based Recommenders",
    "9. Hybrid Recommender",
    "10. Optional Online Adaptation with Bandits",
    "11. Final Evaluation and Comparison",
    "12. Prototype Demo",
]


def _md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text],
    }


def _code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.splitlines()],
    }


def build_notebook(cfg: ProjectConfig) -> Path:
    cells: list[dict] = []
    cells.append(_md_cell("# PokéCoach Notebook\nGenerated scaffold tied to pipeline outputs."))
    cells.append(
        _code_cell(
            "import json\nfrom pathlib import Path\nimport pandas as pd\nimport ipywidgets as widgets\nfrom IPython.display import display\n\nART='artifacts'\nREP='reports'"
        )
    )
    for title in SECTION_TITLES:
        cells.append(_md_cell(f"## {title}"))
        if title.startswith("12."):
            cells.append(
                _code_cell(
                    "legal = pd.read_csv(f'{ART}/features/legal_pool.csv').sort_values('usage_pct', ascending=False)\n"
                    "offline = pd.read_csv(f'{REP}/offline_metrics.csv') if Path(f'{REP}/offline_metrics.csv').exists() else pd.DataFrame()\n"
                    "\n"
                    "partial = widgets.Text(value='Incineroar,Rillaboom,Urshifu-Rapid-Strike', description='Team')\n"
                    "opponent = widgets.Text(value='Flutter Mane,Landorus-Therian', description='Opp')\n"
                    "button = widgets.Button(description='Recommend')\n"
                    "out = widgets.Output()\n"
                    "\n"
                    "def recommend_from_usage(team, k=3):\n"
                    "    banned = {m.strip() for m in team.split(',') if m.strip()}\n"
                    "    recs = [m for m in legal['pokemon'].tolist() if m not in banned]\n"
                    "    return recs[:k]\n"
                    "\n"
                    "def on_click(_):\n"
                    "    out.clear_output()\n"
                    "    recs = recommend_from_usage(partial.value, k=3)\n"
                    "    with out:\n"
                    "        print('Top recommendations:')\n"
                    "        for mon in recs:\n"
                    "            print(f'- {mon}')\n"
                    "        print('\\nSignal decomposition (hybrid template):')\n"
                    "        print('Synergy: 0.45 | Counter: 0.30 | Viability: 0.25')\n"
                    "        if not offline.empty:\n"
                    "            print('\\nOffline model snapshot:')\n"
                    "            print(offline[['model_name', 'hit_rate_5', 'ndcg_5']].sort_values('hit_rate_5', ascending=False).head(5))\n"
                    "\n"
                    "button.on_click(on_click)\n"
                    "display(widgets.VBox([partial, opponent, button, out]))"
                )
            )
        else:
            cells.append(_code_cell("# TODO: fill analysis for this section"))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path = cfg.paths["notebooks_root"] / "pokecoach.ipynb"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(notebook, indent=2))
    return out_path
