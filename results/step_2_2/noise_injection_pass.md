# Noise injection — Pass@4 (exact vs randomized leverage)

Checkpoint **300**, **200** problems, top-4 among alive, `proj_dim=128`, `k=8`.

- **Projection MC:** 64 seeds averaged per problem per trial.
- **Trials:** 40 independent runs (new noise for η>0; η=0 varies projection seeds only).
- **SE:** standard error of the *trial-level* mean Pass@4 (std / sqrt(trials)).

| η | Pass@4 exact | Pass@4 random | random−exact (pp) |
|---|---:|---:|---:|
| 0.0 | 73.50 ± 0.00\% | 72.88 ± 0.01\% | -0.62 ± 0.01 |
| 0.02 | 72.74 ± 0.07\% | 72.78 ± 0.02\% | +0.04 ± 0.06 |
| 0.05 | 72.92 ± 0.12\% | 72.67 ± 0.02\% | -0.25 ± 0.12 |
| 0.1 | 73.00 ± 0.12\% | 72.61 ± 0.02\% | -0.39 ± 0.11 |
| 0.2 | 72.64 ± 0.15\% | 72.66 ± 0.03\% | +0.02 ± 0.13 |

If randomized were a superior noise filter, random Pass@4 would degrade more slowly than exact as η increases. Typically both drop; if random drops faster or tracks exact, do not claim denoising—only JL approximation + cost.
