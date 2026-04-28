# Oracle Pass@m — alive branch policy (Step 2.1)

- Selection uses only branches with `alive_mask[checkpoint][b] == True`.
- Effective `m_eff = min(m, num_alive)`. Retained set size is `m_eff` (all alive if fewer than `m`).
- If `num_alive == 0`, the (problem, checkpoint) pair is excluded from aggregates unless `--include-no-alive` is set.

## Two metrics (do not conflate)

- **Omniscient (ceiling)** — `method="omniscient"` in aggregates: `1` iff at least one **alive** branch has a correct final answer (same graded labels as Step 1). This is the **policy-free** upper bound for “keep \(m\) branches and want ≥1 correct”: if you could choose which branches to keep with full knowledge of correctness, you would succeed whenever such a branch exists among alive. **Does not depend on leverage or logprob.** Implemented as `omniscient_retention_pass` in `src/step2_metrics.py`. Rows are duplicated per \(m\) with the same value (ceiling depends only on alive/correct, not on \(m\)).

- **Policy retention Pass@m** (historically called “Oracle Pass@m” in CSV): `1` iff at least one **correct** branch appears in the **retained** top-\(m\) under the rule (**random** / **logprob** / **exact leverage**). This is **≤** the omniscient rate.

- **Logprob:** sort alive branches by `checkpoint_cumulative_logprob[checkpoint]` descending; take top `m_eff`.
- **Exact leverage:** SVD on rows `H[alive]`; top `m_eff` by `exact_row_leverage_scores` (see `src/leverage.py`).
- **Random:** `random_trials` independent samples of `m_eff` distinct alive branches (uniform without replacement); reported value is the fraction of trials with ≥1 correct branch.

Correctness comes from `solution_texts/branch_*.txt` vs reference (`boxed_only` by default).
