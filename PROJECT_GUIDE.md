# Project Guide

This guide keeps the repository runnable with a minimal workflow.

## 1) Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2) Data

- Place the MATH-500 jsonl file at the path expected by `config/defaults.py`.
- Keep local datasets and large model caches out of git.

## 3) Main Experiments

- Coverage study:
  - `python experiments/step2_1_coverage.py --limit 200 --checkpoints 300 --m 1 2 4 8`
- Approximation study:
  - `python experiments/step2_2_approximation.py --checkpoint 300 --limit 200 --mode all --num_pass_draws 50`
- Schedule sweep:
  - `python experiments/step2_3_schedule_sweep.py --limit 200 --t1-list 50 100 150 --t2-list 50 100 200 300 400 500`

## 4) Outputs

- Core outputs are written under `results/step_2_1`, `results/step_2_2`, and `results/step_2_3`.
- Regenerated outputs may differ slightly when randomized methods are used.

## 5) Repository Policy

- Commit source code, configs, and concise docs.
- Do not commit local caches, zipped dumps, or large generated intermediate artifacts.
