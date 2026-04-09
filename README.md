# PokéCoach

Kickoff implementation for a context-aware Pokemon VGC recommender with offline metrics and simulation-backed evaluation.

## Quick Start

1. Install dependencies:
   - `make setup`
2. Download raw datasets and parse:
   - `make download`
3. Build preprocessing artifacts:
   - `make preprocess`
4. Evaluate all recommenders offline:
   - `make evaluate`
5. Run simulation (Reg G smoke):
   - `make simulate`
6. Run hyperparameter tuning (generates `reports/tuning_*.csv`):
   - `make tune`
7. Generate demo payload + notebook scaffold:
   - `make demo-data`
   - `make notebook`

> **Note:** Step 6 (`make tune`) must run before opening the notebook — the tuning
> visualisation cells (Section 12) load `reports/tuning_hybrid_weights.csv` and
> related files that only exist after this step.

## Instant Frontend Test

To quickly test the frontend without running the full pipeline:

1. Install dependencies:
   - `make setup`
2. Start the Streamlit app:
   - `streamlit run app.py`
3. Open the local URL shown in your terminal (usually `http://localhost:8501`).

## Pipeline Commands

```
python3 -m pokecoach.cli download
python3 -m pokecoach.cli preprocess
python3 -m pokecoach.cli evaluate
python3 -m pokecoach.cli simulate --mode smoke --tier gen9vgc2024regg
python3 -m pokecoach.cli simulate --mode dev   --tier gen9vgc2024regg
python3 -m pokecoach.cli simulate --mode final --tier gen9vgc2024regg
python3 -m pokecoach.cli tune
python3 -m pokecoach.cli demo-data
python3 -m pokecoach.cli make-notebook
```

### File dependency order

| Step | Command | Outputs needed by |
|---|---|---|
| download | `make download` | preprocess |
| preprocess | `make preprocess` | evaluate, tune, notebook |
| evaluate | `make evaluate` | notebook (Section 5, 11) |
| simulate | `make simulate` | notebook (Section 11.1) |
| **tune** | **`make tune`** | **notebook (Section 12)** |
| demo-data | `make demo-data` | notebook (Section 15) |

## Showdown Setup (for real battles)

If you want real local battles rather than fallback simulation:

1. Install Node.js.
2. Run `scripts/setup_showdown.sh`.
3. In another terminal run `scripts/run_showdown.sh`.
4. Re-run simulation commands.

When Showdown is unavailable, the simulation runner keeps the pipeline unblocked with a local fallback so reports are still generated.
