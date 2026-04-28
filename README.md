# Leverage-Based Branch Pruning

This repository contains code for post-hoc analysis of branch pruning strategies on MATH-500 style reasoning traces.

## Project Layout

- `src/`: reusable library code (leverage, metrics, data loading)
- `experiments/`: runnable scripts for each experiment step
- `config/`: default configuration values
- `data/`: input dataset files (not all data is committed)
- `results/`: generated outputs from experiments

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

Run core analysis scripts from the project root:

```bash
python experiments/step2_1_coverage.py --limit 200 --checkpoints 300 --m 1 2 4 8
python experiments/step2_2_approximation.py --checkpoint 300 --limit 200 --mode all --num_pass_draws 50
python experiments/step2_3_schedule_sweep.py --limit 200 --t1-list 50 100 150 --t2-list 50 100 200 300 400 500
```

## Notes

- Large generated artifacts and local model caches are intentionally ignored by `.gitignore`.
- If you need full reproducibility outputs, regenerate them by re-running the scripts above.
