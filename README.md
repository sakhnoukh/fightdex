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
6. Generate demo payload + notebook scaffold:
   - `make demo-data`
   - `make notebook`

## Pipeline Commands

- `python3 -m pokecoach.cli download`
- `python3 -m pokecoach.cli preprocess`
- `python3 -m pokecoach.cli evaluate`
- `python3 -m pokecoach.cli simulate --mode smoke --tier gen9vgc2024regg`
- `python3 -m pokecoach.cli simulate --mode dev --tier gen9vgc2024regg`
- `python3 -m pokecoach.cli simulate --mode final --tier gen9vgc2024regg`
- `python3 -m pokecoach.cli demo-data`
- `python3 -m pokecoach.cli make-notebook`

## Showdown Setup (for real battles)

If you want real local battles rather than fallback simulation:

1. Install Node.js.
2. Run `scripts/setup_showdown.sh`.
3. In another terminal run `scripts/run_showdown.sh`.
4. Re-run simulation commands.

When Showdown is unavailable, the simulation runner keeps the pipeline unblocked with a local fallback so reports are still generated.
