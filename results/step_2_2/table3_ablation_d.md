# Table 3 — Ablation on d (tab:ablation-d)

**Omniscient Pass@4 (ceiling):** 76.50% (n=200) — any correct among alive branches; same `alive` policy as policies below.

**Policy Pass@4 (randomized leverage):** mean over 50 independent projection seeds per (problem, checkpoint), then mean over problems. Column `oracle_pass_at_4_mean` duplicates `policy_pass_at_4_mean` in JSON for backward compatibility.

| Dimension | Mean Kendall-τ | Policy Pass@4 | Time (ms) |
|-----------|----------------|---------------|-----------|
| d=8 | 0.2371 | 72.43% | 0.2022 |
| d=16 | 0.3483 | 72.49% | 0.2693 |
| d=32 | 0.4542 | 72.70% | 0.4832 |
| d=64 | 0.5512 | 72.89% | 0.7333 |
| d=128 | 0.6411 | 72.94% | 1.2848 |
| d=256 | 0.7233 | 72.86% | 3.3008 |
| d=512 | 0.7870 | 72.91% | 4.2441 |
| d=1024 | 0.8371 | 72.98% | 8.0725 |
| Exact SVD (D=1536) | 1.0000 | 73.50% | 1.1589 |
